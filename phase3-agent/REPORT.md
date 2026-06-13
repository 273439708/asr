# 智能座舱语音助手端到端 Demo — 设计与实测报告

> Phase 3 · 车控指令音频 → 端侧 Qwen3-ASR → LangGraph Agent（云端意图识别 + function calling）→ 模拟车控 → 自然语言回复。
> 全链路本机实跑，结果可按文末命令复跑。
> 环境：Windows x64 · Python 3.11 · langgraph 1.0.4 · langchain-openai 1.1.0 · onnxruntime-gpu 1.26.0 · edge-tts 7.2.8

## 0. 诚实说明（先讲清楚边界）

本阶段验证的是 **「端侧 ASR → Agent 工具调用」整条链路能跑通**，不是 ASR 在真实
车内环境下的识别精度：

- **指令音频是 TTS 合成的**（微软 edge-tts，`zh-CN-XiaoxiaoNeural`），清晰普通话、
  无噪声、无口音。Phase 1/2 用的 AISHELL-1 是新闻朗读语料，没有车控指令句，所以
  这里自造 8 条覆盖车控/导航/媒体三域的指令。真实车内的远场、胎噪、方言不在本次评测范围。
- **车控工具是内存里的状态机**（`agent/tools.py` 的 `VEHICLE` dict），不连任何真实
  车机 / CAN 总线，调用即改 dict 并返回中文确认串。
- **意图识别走云端 API**（deepseek，OpenAI 兼容接口）——这一段不是端侧。端侧的是
  ASR；意图 LLM 放云端是工程上的常见分工（语音转写在端、语义理解在云）。

小结：这是一个**端侧 AI 个人项目**，证明的是"能把端侧 ASR 模型接进一个 Agent 闭环并正确
驱动工具"，不是"做出了量产车机语音助手"。

## 1. 架构

```
                 commands/*.wav (16kHz mono, TTS 合成)
                          │
              ┌───────────▼───────────┐
              │  asr_node              │  端侧：复用 Phase 2 的
              │  Qwen3-ASR-0.6B ONNX   │  QwenASROnnxPipeline（int8）
              │  (onnx_int8)           │  wav → 识别文本
              └───────────┬───────────┘
                          │ transcript
              ┌───────────▼───────────┐
              │  agent_node            │  云端：deepseek bind_tools
              │  deepseek + 8 tools    │  文本 → tool_calls 或直接回复
              └─────┬──────────────┬───┘
              有 tool_calls      无 tool_calls
                    │                │
        ┌───────────▼──────┐         │
        │  tools_node      │         │
        │  执行车控并改     │         │
        │  VEHICLE 状态     │         │
        └───────────┬──────┘         │
                    │ ToolMessage    │
                    └──► agent_node ──┘  （ReAct 闭环，回去拿自然语言回复）
                                     │
                                  final_reply
```

- **框架**：LangGraph `StateGraph`，状态 `CockpitState`（TypedDict，`messages` 用
  `add_messages` reducer）。`agent → tools → agent` 条件边构成 ReAct 循环，无 tool_calls 即 END。
- **ASR 复用**：`agent/asr.py` 是 Phase 2 `export/onnx_pipeline.py` 的薄封装，直接
  加载 `phase2-qwen-asr/onnx_int8/` 的 5 个 ONNX，不重新导出。
- **工具集**（`agent/tools.py`，8 个 `@tool`，JSON Schema 由类型注解 + docstring 自动生成）：
  车控 `set_climate_temperature` / `control_window` / `set_seat_heating`；
  导航 `navigate_to` / `find_nearby`；媒体 `media_play` / `set_media_volume` / `media_pause`。

## 2. 8 条指令实测（端到端，一次跑通）

| # | 音频 | ASR 识别文本 | 工具调用 | 车辆状态变化 | 评判 |
|---|---|---|---|---|---|
| 01 | climate | 把空调调到二十六度。 | `set_climate_temperature(26)` | climate_temp 24→26 | ✅ |
| 02 | window | 打开主驾驶车窗。 | `control_window(主驾, 100)` | windows.主驾 0→100 | ✅ |
| 03 | seat | 副驾座椅加热开到三档。 | `set_seat_heating(副驾, 3)` | seat_heat.副驾 0→3 | ✅ |
| 04 | nav | 导航到上海虹桥火车站。 | `navigate_to(上海虹桥火车站)` | nav_destination → 上海虹桥火车站 | ✅ |
| 05 | poi | 找一下附近的充电站。 | `find_nearby(充电站)` | （只读，无状态变化） | ✅ |
| 06 | media | 播放周杰伦的《稻香》。 | `media_play(周杰伦 稻香)` | media_playing → 周杰伦 稻香 | ✅ |
| 07 | multi | 有点热，把空调降到二十二度，再把天窗打开一半。 | `set_climate_temperature(22)` | climate_temp 24→22 | ✅* |
| 08 | chat | 今天天气怎么样？ | （无） | （无） | ✅ |

**8/8 全部符合预期。** ASR 识别全部逐字正确（含「稻香」「上海虹桥火车站」等专名）。

两个值得说的 case：

- **07 多意图 + 能力边界**：一句话两个意图。空调降 22 度正确执行；「天窗」不在工具集
  （只支持主驾/副驾/左后/右后四个车窗），LLM **没有幻觉调用**，而是诚实回复"暂时不支持
  控制天窗"。这正是想验证的——意图超出工具能力时优雅降级，而不是瞎调一个工具。
- **08 无关指令**：天气查询不属于车载域，LLM 不调任何工具，直接说明能力边界。

ASR 单段耗时（onnx_int8，CPU EP）：1.8–3.3s/条，随音频长度增长（07 最长 3.32s）。

## 3. 关键实现点

1. **ASR 输出截断**：Qwen3-ASR 转写结果在 `<asr_text>` token（id 151704）之后才是正文，
   `body = toks[toks.index(ASR_TEXT)+1:]` 取正文再 decode。
2. **温度采样 = 0**：意图 LLM `temperature=0`，保证「同一句指令 → 同一个工具调用」稳定可复现。
3. **System prompt 划定边界**：明确"只处理车控/导航/媒体，可调多个工具，非车载请求礼貌拒绝"，
   是 07/08 能正确降级的关键。
4. **Windows 控制台 UTF-8**：`sys.stdout.reconfigure(encoding="utf-8")`，否则 GBK 编码不了
   中文与 ♪ 等符号。

## 4. 复跑命令

```bash
cd phase3-agent

# 1) 配密钥（填入 OpenAI 兼容云 API 的 deepseek key）
cp .env.example .env && 编辑填入 OPENAI_API_KEY

# 2) 生成指令音频（edge-tts，需联网）
../phase2-qwen-asr/venv/Scripts/python make_commands.py

# 3) 端到端 demo（全部 8 条 / 指定单条）
../phase2-qwen-asr/venv/Scripts/python main.py
../phase2-qwen-asr/venv/Scripts/python main.py 07_multi
```
