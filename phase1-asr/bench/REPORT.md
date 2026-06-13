# Whisper-base ONNX 端侧部署 — 量化与性能报告

> Phase 1 ASR · 导出 / 量化 / 评测全链路，所有数字均为本机实测（可按文末命令复跑）。
> 环境：Windows x64 · Python 3.11 · torch 2.9.1 · transformers 4.57.5 · onnxruntime 1.20.1（CPU EP）· opset 17

## 1. 模型拆分与导出

PyTorch → ONNX（TorchScript 跟踪导出，`dynamo=False`），拆成 4 个模型：

| 模型 | 作用 | fp32 | int8 | 压缩比 |
|---|---|---:|---:|---:|
| encoder.onnx | mel(80×3000) → hidden | 82.4 MB | 26.0 MB | 32% |
| decoder_no_cache.onnx | 全序列重算解码（O(n²) 基线） | 314.4 MB | 79.5 MB | 25% |
| cross_kv_init.onnx | cross-attn K/V 一次性预计算 | 12.6 MB | 3.2 MB | 26% |
| decoder_with_cache.onnx | 增量解码（self-attn KV cache） | 301.8 MB | 76.3 MB | 25% |

导出期解决的两个兼容问题（细节见 `export/export_whisper_onnx.py` 注释）：

1. **vmap mask 不兼容**：transformers≥4.53 的 `masking_utils` 用 `torch.vmap` 构建因果
   mask，TorchScript 跟踪不支持（报 `invalid unordered_map<K, T> key`）。用纯广播实现
   替换注册表中的 sdpa/eager 两个 mask 函数（导出场景无 padding，语义等价）。
2. **cross-attn 被静默剪枝**：whisper decoder 层以 `encoder_hidden_states is not None`
   决定是否执行 cross-attn 块。带 cache 导出时不传该参数 → cross-attn 整块被跳过、
   cross_k/v 输入被剪掉，模型"成功导出"但完全不依赖音频。通过检查导出模型的输入列表
   发现该问题，修复方式是从 cross_k 还原形状传一个占位张量（`is_updated=True` 时内容
   不参与计算）。

**fp32 数值对齐**（ONNX vs PyTorch, max abs diff）：encoder 1.4e-03 ·
decoder_no_cache 1.8e-04 · cross_kv 6.4e-04 · decoder_with_cache logits 2.2e-05。
贪心 64 步 token 序列与 PyTorch 完全一致。

## 2. KV cache 收益（fp32, ONNX Runtime CPU）

贪心解码 64 步，同一 encoder 输出：

| 解码方式 | ms/step | 加速比 |
|---|---:|---:|
| 无 cache（每步重算全序列） | 30.7 | 1.00x |
| 带 KV cache（每步 1 token） | 13.1 | **2.34x** |

token 序列完全一致。无 cache 路径每步代价随序列长度增长，64 步只是下限——序列越长差距越大。

## 3. 动态 INT8 量化

`onnxruntime.quantization.quantize_dynamic`，最终配置：

```python
quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8,
                 op_types_to_quantize=["MatMul", "Gemm", "Gather"],
                 per_channel=True)
```

配置是逐项实验定下的：

- **不量化 Conv**：ORT CPU EP 没有 ConvInteger 实现，默认配置量化 encoder 前端
  2 个 Conv 后直接 `NOT_IMPLEMENTED` 崩溃 → 显式白名单 op 类型，Conv 保持 fp32。
- **量化 Gather**：命中 token embedding 权重表，decoder 体积 156.6→76.3 MB（再减半），
  实测贪心 token 仍全一致。
- **per_channel=True**：encoder 输出 mean|diff| 0.128→0.076，余弦相似度 0.9936→0.9978。

**量化精度**（固定输入，fp32 vs int8）：encoder 最终输出 max diff ~0.5（幅值 ~1.4），
decoder logits argmax 一致；贪心 64+4 步 token 一致率 **68/68（完全一致）**。

**逐层误差定位**（`--layerwise`，把每层残差出口挂成图输出）：

| decoder layer | max diff | mean diff |
|---|---:|---:|
| 0 | 2.1e-02 | 4.4e-03 |
| 1 | 6.5e-02 | 9.8e-03 |
| 2 | 1.4e-01 | 1.9e-02 |
| 3 | 3.2e-01 | 3.6e-02 |
| 4 | 6.1e-01 | 5.7e-02 |
| 5 | 1.1e+00 | 8.0e-02 |

误差逐层单调放大（每层约 ×1.5-2），是典型的量化误差沿残差流累积；最终 logits 误差
未改变 argmax 排序，因此解码结果不受影响。

## 4. 端到端评测（AISHELL-1 test 子集，200 条 / 16.6 分钟音频）

数据：`AudioLLMs/aishell_1_zh_test` parquet 分片（2307 条中 seed=0 随机抽 200）。
指标为中文 CER（按字）；文本规整：繁→简（whisper-base 中文常出繁体）+ 去标点。
管线：encoder → cross_kv_init → decoder_with_cache 贪心。

| | CER | RTF | 首 token 延迟 | 解码 ms/step |
|---|---:|---:|---:|---:|
| fp32 | 21.32% | 0.118 | 389 ms | 14.3 |
| int8 | **22.12%** | **0.080** | **286 ms** | 10.1 |

- **CER 量化损失 +0.80pp**（21.32→22.12%），换体积 711→185 MB（26%）和 RTF 1.48x。
- CER 绝对值符合 whisper-base 在 AISHELL-1 的公开水平（无 LM、贪心、74M 小模型；
  量化前后对比才是本报告的目的）。
- RTF 0.08-0.12 ≪ 1，CPU 实时率充裕。

## 5. 复跑命令

```bash
cd export
python export_whisper_onnx.py --model openai/whisper-base --out ../onnx
python verify_onnx.py --onnx ../onnx --steps 64          # PyTorch 对齐 + KV cache 对比

cd ../quantize
python quantize_int8.py --onnx ../onnx --out ../onnx_int8 --layerwise

cd ../bench
python bench_runtime.py --onnx ../onnx --int8 ../onnx_int8   # RTF/首token（无数据集）
python bench_wer.py --models ../onnx ../onnx_int8 --num 200  # CER 对比
```

依赖数据文件（bench/ 下）：`whisper_fe/`、`whisper_tok/`（hf-mirror 下载的
feature extractor / tokenizer 配置）、`../data/aishell1_test-00000.parquet`。
HF 不可达环境需 `HF_HUB_OFFLINE=1`。
