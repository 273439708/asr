"""座舱 Agent 共享状态。messages 用 add_messages reducer 累积对话/工具消息。"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class CockpitState(TypedDict, total=False):
    audio_path: str          # 输入车控指令音频
    transcript: str          # Qwen3-ASR 识别出的指令文本
    messages: Annotated[list, add_messages]  # system/user/ai/tool 消息流
    tool_calls_log: list     # [(工具名, 入参, 返回)]，演示用执行轨迹
    final_reply: str         # 给用户的最终自然语言回复
