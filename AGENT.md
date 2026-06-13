# AGENT.md — 端侧 ASR 语音助手项目说明

> 面向 AI 助手 / 协作者的项目上下文文档。状态：**三阶段全部完成（阶段 1 ASR / 阶段 2 小 LLM / 阶段 2.5 TensorRT / 阶段 3 Agent 闭环）**。

## 项目是什么

一个分三阶段的端侧 AI 个人学习项目：从 ASR 语音输入起步，逐步串成
「语音输入 → 端侧 ASR → 端侧 LLM 意图识别 → Function Calling 工具调用」的微缩版智能座舱语音助手。
重点是亲手走通端侧大模型的完整部署链路——模型导出、量化、加速、Agent 闭环，每个环节都留下可复跑的实测数据。

## 技术目标与能力覆盖

把以下端侧部署关键能力从「原理了解」做成「有实测数据的工程实践」：

| 能力点 | 由哪个阶段覆盖 | 当前状态 |
|---|---|---|
| ONNX Runtime / 模型转换 / 精度对齐 | 阶段 1（ASR） | ✅ whisper-base 拆 4 模型 + Qwen3-ASR 拆 5 模型，对齐 max diff 1e-3 / 7.6e-06 |
| INT8 量化 + 量化前后精度验证（WER/CER） | 阶段 1、2 | ✅ 两遍：whisper WER +0.80pp，Qwen3-ASR CER +0.31pp，含逐层误差定位 |
| 性能压测 / 自动化测试（RTF、首字延迟、内存） | 阶段 1、2 | ✅ 可复跑脚本 + REPORT（RTF/首token/CER 四件套） |
| 端侧部署（Android） | 阶段 1（ASR） | ✅ 已完成并真机验证 |
| KVCache 原理（Whisper decoder 实测） | 阶段 1 入门，阶段 2 深化 | ✅ whisper 带/不带 cache 2.34x；Qwen3 prefill/增量一图覆盖 |
| INT4 量化（AWQ/GPTQ/llama.cpp k-quants） | 阶段 2（小 LLM） | ⬜ 未开始（阶段 2 用 INT8 路线完成，INT4 为可选补充） |
| KVCache 深度优化 / 首 token 延迟 / 多请求并发 | 阶段 2（小 LLM） | ◐ 首token 251ms 已实测；并发吞吐未做 |
| 端侧 Agent 框架 / 工具调用执行闭环 / 多模态语音助手 | 阶段 3（Agent 闭环） | ✅ LangGraph ReAct 闭环：端侧 ASR→LLM 意图→8 工具，8/8 指令正确（含多意图/越界拒绝） |
| TensorRT | 本机有 NVIDIA 显卡，路线可做 | ✅ TRT 11 FP16 engine：音频编码器 vs ORT CUDA 2.7-4.3x / CPU 45-65x，cos≥0.9987 |

## 选型与硬件（已落定）

- **本机有 NVIDIA 显卡**（RTX 3060 Ti 8GB）→ TensorRT 路线可做（ORT TensorRT EP / FP16/INT8 对比，排期在导出/量化链路之后）
- **阶段 1 选型：sherpa-onnx + Zipformer**（工程最短路径，已落地）；Whisper PC 端链路作为导出/量化/KVCache 练习的补充项
- **Android 真机：小米 11 Pro**（M2102K1AC，Snapdragon 888，arm64-v8a，Android 14/API 34，11GB RAM，无线 ADB）
- **语种：中英文混合** → 模型选用 sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

## 三阶段规划

### 阶段 1：ASR 端侧部署

**✅ 已完成（Android demo，代码在 `M:\projects\AsrDemo`）：**

- sherpa-onnx v1.13.2 官方 AAR + 双语 Zipformer transducer（int8 encoder/joiner + fp32 decoder），CPU 推理
- 麦克风流式识别全链路：AudioRecord(16kHz) → 流式解码 → endpoint 断句 → Compose UI 实时上屏
- 文本后处理：英文大小写规范化、中英边界空格、全角/半角标点适配
- 标点恢复：CT-Transformer zh-en int8（75MB），仅对定稿句子推理
- 实时 RTF 统计显示、识别时屏幕常亮
- **真机实测数据**（Mi 11 Pro，2 线程 CPU）：RTF 0.152（纯 ASR）/ 0.187（含标点），进程 ~328MB PSS，APK 244MB；中英混说、问号判断均验证通过
- 工程细节、踩坑记录、模型获取方式见 `M:\projects\AsrDemo\AGENT.md`

**✅ 模型部署链路（PC 侧，`phase1-asr/`）：**

- [x] PyTorch → ONNX 导出 + 算子兼容处理（`phase1-asr/export`：whisper-base 拆 4 模型，
  解决 vmap mask 不可 trace、cross-attn 被静默剪枝两个兼容问题）
- [x] 自己跑 INT8 量化 + 量化前后逐层精度对齐 + WER 对比（`phase1-asr/quantize`：
  WER 21.32%→22.12%（+0.80pp），体积 711→185MB，逐层误差定位）
- [x] 可复跑 benchmark 脚本（`phase1-asr/bench`：WER/RTF/首token，REPORT.md 含复跑命令）
- [x] Whisper decoder KVCache 带/不带实测（2.34x，64 步贪心序列一致）

**demo 自身的体验向待办**（优先级低）：热词增强（modified_beam_search + bpe 词表）、partial 实时标点、release 构建。

### 阶段 2：小 LLM 端侧部署（✅ 已完成，代码在 `phase2-qwen-asr/`，REPORT 见 bench/）

选型：用 **Qwen3-ASR-0.6B**（音频编码器 + 28 层 Qwen3 decoder 的 LLM-based ASR）
一个模型同时覆盖「小 LLM 端侧部署」与 ASR 主线，比单独部署纯文本 LLM 更贴座舱场景。

- 拆 5 模型 ONNX 导出（MRoPE 退化 1D RoPE、手写 decoder 前向绕开 DynamicCache、
  lm_head 独立避 protobuf 2GB、分窗 mask 图外构造）
- fp32 对齐：音频编码器 max diff 7.6e-06，贪心序列与 PyTorch 完全一致
- INT8：3.58GB→0.91GB（25%），CER 3.78%→4.09%（+0.31pp），
  逐层定位到 massive activation 误差注入（layer 2 起 max ~84，沿残差流累积）
- AISHELL-1 200 条同口径：int8 CPU RTF 0.203、首 token 251ms、1010ms/条
- 对照：PyTorch bf16 CUDA 基线 CER 3.71% / whisper-base int8 22.12%（代差级领先）
- 未做（可选补充）：INT4（AWQ/GPTQ/k-quants）、多请求并发吞吐

### 阶段 2.5：TensorRT（✅ 已完成，代码在 `phase2-qwen-asr/trt/`）

- tensorrt-cu12 **11.0**（pip wheel，RTX 3060 Ti 8GB）：音频编码器两模型 → FP16 engine
- TRT 11 移除弱类型精度 flag → onnxconverter-common 转 fp16 ONNX + 强类型构建
  （另踩 EXPLICIT_BATCH 移除、IHostMemory 无 len() 两个 API 变化）
- 动态形状 profile（N=1/8/64，S=13/104/832 + attn_mask 同步）；mask 用 -1e4 防 fp16 溢出
- 实测：TRT FP16 vs ORT CUDA fp32 **2.7-4.3x**、vs ORT CPU fp32 **45-65x**；
  精度 max diff ~1e-2、cos≥0.99869（好于 int8 编码器的 0.975-0.991）
- 详见 `phase2-qwen-asr/bench/REPORT.md` 第 4 节；decoder 的 TRT 化（KV cache 动态轴）留作可选

### 阶段 3：Agent 闭环（✅ 已完成，代码在 `phase3-agent/`，REPORT 见同目录）

- 链路：车控指令音频 → 端侧 Qwen3-ASR（复用阶段 2 onnx_int8）→ LangGraph ReAct 闭环
  （云端 LLM 意图识别 + function calling）→ 执行 8 个模拟车控工具 → 自然语言回复
- 框架：LangGraph `StateGraph`（`agent↔tools` 条件边构成 ReAct 循环），意图 LLM `temperature=0`
- 8 条 TTS 合成指令（edge-tts，覆盖车控/导航/媒体三域）**8/8 全部正确**：
  - ASR 逐字识别正确（含「稻香」「上海虹桥火车站」等专名），int8 CPU 1.8–3.3s/条
  - 多意图（07）：空调降 22 度执行 + 「天窗」越界**诚实拒绝不幻觉调用**
  - 无关指令（08）：天气查询**正确拒绝**并说明能力边界
- **诚实边界**（写进 REPORT 第 0 节）：指令音频是 TTS 合成的清晰普通话（非真实车内远场/噪声）；
  车控是内存状态机非真实车机；意图 LLM 走云端（端侧的是 ASR）。验证的是**链路打通**非量产精度。
- 复用既有 LangGraph / Function Calling 经验落地工具调用闭环

## 工程诚实原则（不可违反）

- **数字必须实测**：所有指标（WER/CER 损失、RTF、首 token 延迟、加速比等）必须是本项目本机实测产出，不得编造或外推
- **能力点做完才记录**：未完成阶段对应的能力点，在文档中保持「未开始 / 补齐中」表述，不得提前标 ✅
- **benchmark 必须可复跑**：脚本 + 固定测试集 + 环境记录三件套齐全，数据要经得起复现
- **演示场景如实说明**：TTS 合成音频、模拟工具、云端意图识别等非真实/非端侧的部分，须在 REPORT 中明确标注

## 目录现状

```
F:\work\project\ASR\          # 本规划目录（git 仓库）
├── AGENT.md                  # 本文档（总规划与进度跟踪）
├── phase1-asr\               # ✅ whisper-base：export / quantize / bench + REPORT.md
├── phase2-qwen-asr\          # ✅ Qwen3-ASR-0.6B：export / quantize / bench + REPORT.md
│   ├── onnx\ onnx_int8\      #    模型产物（gitignore，按 REPORT 复跑命令再生成）
│   ├── trt\ trt_plans\       #    ✅ 阶段 2.5 TensorRT（plans gitignore）
│   └── venv\                 #    独立 venv（qwen-asr + onnxruntime，torch 复用系统包）
└── phase3-agent\             # ✅ 阶段 3：agent/(asr/llm/tools/graph/state) + main.py + REPORT.md
    ├── make_commands.py      #    edge-tts 合成 8 条车控指令 wav（gitignore）
    └── .env                  #    云端 API 密钥（gitignore，从 .env.example 复制）

M:\projects\AsrDemo\          # 阶段 1 Android demo（独立 git 仓库）
├── AGENT.md                  # demo 工程文档：版本栈/命令/架构/踩坑/模型获取
├── app/src/main/java/com/example/asrdemo/
│   ├── MainActivity.kt       # Compose UI + AudioRecord + 工作线程
│   ├── SherpaAsrEngine.kt    # OnlineRecognizer(ASR) + OfflinePunctuation(标点)
│   └── TextPostProcessor.kt  # 大小写/空格/标点润色
└── models\                   # 下载缓存 ~700MB（可删，已 gitignore）

模型权重：M:\models\Qwen3-ASR-0.6B（modelscope 下载）
```

## 当前状态与待办

- [x] 立项：三阶段规划、能力点映射确定
- [x] 落定硬件 / 选型 / 真机 / 语种
- [x] 阶段 1 Android demo 全链路跑通并真机验证（识别 + 标点 + RTF + 后处理）
- [x] 阶段 1 导出/量化/精度对齐/benchmark 全链路（whisper-base）
- [x] 阶段 2 Qwen3-ASR-0.6B 全链路（REPORT 见 `phase2-qwen-asr/bench/REPORT.md`）
- [x] 阶段 2.5 TensorRT：TRT 11 FP16 engine vs ORT CPU/CUDA 延迟对比（REPORT 第 4 节）
- [x] 阶段 3 Agent 闭环：LangGraph ReAct（端侧 ASR→LLM 意图→8 工具），8/8 指令正确（REPORT 见 `phase3-agent/REPORT.md`）
- [ ] 可选补充：INT4（AWQ/GPTQ/k-quants）、并发吞吐、Qwen3-ASR int8 上 Android 真机

## 关联资料

- 阶段 1 demo 工程文档：`M:\projects\AsrDemo\AGENT.md`
