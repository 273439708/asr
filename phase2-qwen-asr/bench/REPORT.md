# Qwen3-ASR-0.6B ONNX 端侧部署 — 导出、量化与评测报告

> Phase 2 · 0.6B LLM-based ASR 的 PyTorch → ONNX 全链路，所有数字均为本机实测（可按文末命令复跑）。
> 环境：Windows x64 · Python 3.11 · torch 2.9.1 · transformers 4.57.6 · qwen-asr 0.0.6 · onnxruntime 1.20.1（CPU EP）· opset 17
> 对照基线：Phase 1 whisper-base（74M encoder-decoder）同口径 CER 21.32%。

## 1. 模型结构与导出拆分

Qwen3-ASR-0.6B = 18 层音频编码器（conv×3 + 双向 transformer, 896→1024）+ 28 层 Qwen3
decoder（GQA 16Q:8KV, head_dim 128, q/k per-head RMSNorm）。与 whisper 的本质区别：
**没有 cross-attention**——音频嵌入（约 13 token/秒）直接替换 prompt 中的
`<|audio_pad|>` 占位符，之后就是普通 LLM 自回归解码。

TorchScript 跟踪导出（`dynamo=False`），拆 5 个模型：

| 模型 | 作用 | fp32 | int8 | 压缩比 |
|---|---|---:|---:|---:|
| audio_frontend.onnx | mel 块[N,1,128,100] → [N,13,896]（conv×3 + 块内位置编码） | 42.1 MB | 22.5 MB | 53% |
| audio_transformer.onnx | [S,896] + 分窗 mask → 音频嵌入 [S,1024] | 669.0 MB | 168.8 MB | 25% |
| embed.onnx | token id → 嵌入（Gather 查表） | 593.5 MB | 148.4 MB | 25% |
| decoder.onnx | 28 层 + KV cache，prefill/增量解码共用一张图 | 1680.8 MB | 422.9 MB | 25% |
| lm_head.onnx | hidden[1,1,1024] → logits[1,1,151936] | 593.5 MB | 149.1 MB | 25% |
| **合计** | | **3.58 GB** | **0.91 GB** | **25%** |

导出期的四个关键决策（细节见 `export/export_qwen3_asr_onnx.py` 注释）：

1. **MRoPE 退化为 1D RoPE**：读源码确认 ASR 场景下 `get_rope_index` 给 3 路
   （T/H/W）的 position_ids 完全相同，`apply_interleaved_mrope` 是恒等变换 →
   decoder 按标准 RoPE 重写导出，绕开了最大的兼容性风险点。
2. **手写 decoder 前向**：HF 的 `DynamicCache` / `masking_utils` 不可 trace。用原模块
   权重（q_proj/k_norm/gate_proj/...）重写无 Cache 类的前向，position 与因果 mask 从
   `past_k.size(3)` 动态推导——一张图同时覆盖 prefill（L>1, P=0）与增量解码（L=1, P>0）。
3. **lm_head 单独拆出**：① 5.9 亿参数 fp32 decoder 若含 lm_head 超 protobuf 2GB 上限；
   ② 贪心解码只需最后一个位置的 logits，prefill 阶段省掉 L×151936 的大矩阵乘。
4. **分窗注意力 mask 图外构造**：音频编码器 8s 一窗的 block-diagonal mask（含尾块
   有效帧数 `(t-1)//2+1` 三次的逐级推导）在 numpy 侧完成，作为显式输入传图，
   避免 trace 不定形状的 `cu_seqlens` 逻辑。

**fp32 数值对齐**（ONNX vs PyTorch fp32 CPU，3 条不同长度样本）：音频编码器
max abs diff **7.6e-06**；端到端贪心 token 序列**逐 token 完全一致**——同时验证了
块数 N / 序列 S / prefill 长度 L / cache 长度 P 四个动态轴。
一个排查记录：HF `thinker.generate` 默认没把 `<|im_end|>` 当停止符，生成到
max_new_tokens 才停，首轮对比"不一致"实为参考序列未截断，截到首个 EOS 后全对齐。

## 2. 动态 INT8 量化

复用 Phase 1 验证过的配置：

```python
quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8,
                 op_types_to_quantize=["MatMul", "Gemm", "Gather"],
                 per_channel=True)
```

audio_frontend 的 3 个 Conv2d 保持 fp32（ORT CPU EP 无 ConvInteger），所以它压缩比
只有 53%；其余四个模型纯 MatMul/Gather，均压到 25%。

**量化对齐**（真实音频，fp32 vs int8）：音频编码器输出余弦相似度 0.975-0.991；
端到端贪心 3 条样本 1 条逐 token 一致，2 条**仅差一个插入的逗号**（文字内容完全相同）
——0.6B LLM 对权重量化的鲁棒性体现在语义层，标点这类低置信度 token 先翻转。

**逐层误差定位**（`--layerwise`，把每层残差出口挂成图输出，prefill 一步）：

| decoder layer | max diff | mean diff | 隐层幅值 |
|---|---:|---:|---:|
| 0 | 1.9e-01 | 2.8e-02 | 0.20 |
| 1 | 3.0e-01 | 3.7e-02 | 0.22 |
| 2 | **8.4e+01** | 1.2e-01 | 0.40 |
| 9 | 8.4e+01 | 2.0e-01 | 0.59 |
| 18 | 8.2e+01 | 5.8e-01 | 1.53 |
| 27 | 1.3e+03 | 4.0e+00 | 6.74 |

两个现象：① max diff 在 layer 2 突跳到 ~84 后基本不变——量化误差命中了 LLM 残差流中
著名的 massive activation 异常通道（个别坐标幅值远超其余），由残差直通逐层携带；
② mean diff 沿残差流单调累积（×1.1-1.3/层），与隐层幅值同步增长，相对误差稳定。
最终 logits 的 argmax 排序对这种误差不敏感，贪心结果只在标点上抖动。

## 3. 端到端评测（AISHELL-1 test 子集，200 条 / 16.6 分钟音频）

与 Phase 1 完全同口径：同一 parquet、seed=0 抽同 200 条、同 normalize
（繁→简 + NFKC + 去标点）、jiwer.cer。贪心解码，CER 按字。

| | CER | RTF | ms/条 | 首 token 延迟 |
|---|---:|---:|---:|---:|
| PyTorch bf16 CUDA（基线） | 3.71% | 0.112 | 555 | — |
| ONNX fp32 CPU | 3.78% | 0.40-0.64* | 3201 | 872 ms |
| ONNX int8 CPU | **4.09%** | **0.203** | **1010** | **251 ms** |

\* fp32 前 150 条 RTF 稳定在 ~0.40，最后 50 条劣化到累计 0.644——3.6GB 权重的工作集
接近本机内存压力线所致；int8（0.91GB）全程稳定，这本身就是量化的端侧价值之一。

- **导出无损**：ONNX fp32 与 GPU bf16 基线 CER 差 +0.07pp（3.71→3.78%），属
  bf16/fp32 数值路径差异，贪心序列已验证与 fp32 PyTorch 完全一致。
- **量化损失 +0.31pp**（3.78→4.09%），换体积 3.58GB→0.91GB（25%）、单条耗时
  3.2x、首 token 延迟 3.5x。损失主要来自标点插入/删除与少量同音字翻转。
- **与 Phase 1 的精度-成本对照**：whisper-base int8 CER 22.12% / 185MB，
  Qwen3-ASR int8 CER 4.09% / 912MB——5 倍体积换 ~18pp CER，0.6B LLM-ASR 在中文上
  代差级领先；且 RTF 0.2 仍有 5 倍实时余量。

## 4. TensorRT FP16（阶段 2.5 补充，2026-06-13）

> 环境补充：RTX 3060 Ti 8GB · tensorrt-cu12 **11.0.0.114**（pip wheel，无 trtexec）·
> onnxruntime-gpu 1.26.0（CUDA EP）· onnxconverter-common 1.16

**范围**：音频编码器两个模型（frontend + transformer）——计算密集、形状规整，是 TRT
收益最大的部分；28 层 decoder 的自回归 + KV cache 动态轴留作可选补充。

**TRT 11 的三个 API 变化（踩坑记录，网上多数教程还是 10.x 写法）**：

1. **弱类型精度 flag 全部移除**（`BuilderFlag.FP16/INT8/BF16` 等）：网络默认强类型，
   精度跟随模型自身 dtype → 先用 onnxconverter-common 把 fp32 ONNX 转 fp16
   （`keep_io_types=True`，I/O 保持 fp32 由图内 Cast 衔接），再常规构建。
2. `NetworkDefinitionCreationFlag.EXPLICIT_BATCH` 移除 → `create_network(0)`。
3. `IHostMemory` 不再支持 `len()`/切片 → `bytes(blob)` 后再落盘。

**动态形状 optimization profile**：frontend 按块数 N=1/8/64，transformer 按帧数
S=13/104/832（attn_mask 同步 [1,1,S,S]）。两个 engine 构建共约 30s，plan 落盘缓存。
**fp16 数值安全**：分窗注意力的加性 mask 由 fp32 min 改为 **-1e4**（fp32 min 在
fp16 下溢出为 -inf，softmax 全 -inf 行会出 NaN）。

**延迟与精度**（同一真实样本 50 次均值，含 H2D/D2H 拷贝；diff/cos 以 ORT CPU fp32 为基准）：

| mel T | S | ORT CPU fp32 | ORT CUDA fp32 | TRT FP16 | max diff | cos |
|---:|---:|---:|---:|---:|---:|---:|
| 363 | 47 | 179.0 ms | 7.8 ms | **2.9 ms** | 2.6e-02 | 0.99869 |
| 591 | 77 | 229.3 ms | 13.3 ms | **3.3 ms** | 4.9e-03 | 0.99995 |
| 437 | 57 | 149.8 ms | 14.1 ms | **3.3 ms** | 1.2e-03 | 0.99999 |
| 3000（~30s 拼接音频） | 390 | 751.4 ms | 35.8 ms | **11.5 ms** | 8.3e-03 | 0.99995 |

- TRT FP16 比 ORT CUDA fp32 快 **2.7-4.3x**，比 ORT CPU fp32 快 **45-65x**；
  长音频（S=390）下优势依然保持，profile 的 opt 点（S=104，8s 窗）覆盖常见句长。
- FP16 精度损失 max diff ~1e-2 量级、**cos ≥ 0.99869**——好于 int8 量化时编码器的
  余弦（0.975-0.991）。按第 2 节 int8 实验的经验（该级别编码器扰动端到端只翻转标点
  类低置信度 token），FP16 编码器对 CER 的影响应小于 int8 路线（推断，未单独跑 200 条）。

## 5. 复跑命令

```bash
cd export
python export_qwen3_asr_onnx.py --model M:/models/Qwen3-ASR-0.6B --out ../onnx
python verify_onnx.py --onnx ../onnx            # PyTorch 对齐（编码器 diff + 贪心序列）

cd ../quantize
python quantize_int8.py --onnx ../onnx --out ../onnx_int8 --layerwise

cd ../bench
python bench_cer_pytorch.py --num 200           # GPU bf16 基线（需 CUDA）
python bench_cer_onnx.py --onnx ../onnx --tag fp32 --num 200
python bench_cer_onnx.py --onnx ../onnx_int8 --tag int8 --num 200

cd ../trt                                        # TensorRT（需 NVIDIA GPU）
python trt_bench.py --onnx ../onnx --plan ../trt_plans --iters 50
```

依赖：模型权重 `M:/models/Qwen3-ASR-0.6B`（modelscope 下载）、
数据 `../../phase1-asr/data/aishell1_test-00000.parquet`（与 Phase 1 共用）、
venv 见 `pip install qwen-asr onnx onnxruntime jiwer zhconv`（torch 复用系统包）；
TensorRT 部分另需 `pip install tensorrt-cu12 onnxruntime-gpu onnxconverter-common`。
