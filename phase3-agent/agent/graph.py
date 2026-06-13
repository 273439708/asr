"""座舱 Agent LangGraph 图：

    START → asr → agent ──有 tool_call──→ tools ──┐
                  ↑                                 │
                  └─────────────────────────────────┘
                  └──无 tool_call──→ END（final_reply）

asr 节点把音频转写成文本并塞入第一条 user 消息；agent 节点调 deepseek（已 bind_tools）
做意图识别，决定调哪个座舱工具；tools 节点执行工具改 VEHICLE 状态并回灌 ToolMessage；
循环直到 LLM 不再请求工具，输出自然语言回复。ReAct 闭环。
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .asr import CockpitASR
from .llm import get_llm_with_tools
from .state import CockpitState
from .tools import TOOLS_BY_NAME

SYSTEM_PROMPT = (
    "你是智能座舱语音助手。用户的话来自车内语音识别，可能有口语化或识别噪声。"
    "请理解用户意图，调用合适的工具完成车控（空调/车窗/座椅加热）、导航、媒体操作。"
    "可以一次调用多个工具（如同时开两个车窗）。工具执行后，用一句简短自然的中文向用户确认结果。"
    "如果用户的话与车辆操作无关，礼貌说明你只能处理车控/导航/媒体类指令。"
)

_asr = None


def _get_asr() -> CockpitASR:
    global _asr
    if _asr is None:
        _asr = CockpitASR()
    return _asr


def asr_node(state: CockpitState) -> dict:
    text, _ = _get_asr().transcribe(state["audio_path"])
    return {
        "transcript": text,
        "messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(text)],
    }


def agent_node(state: CockpitState) -> dict:
    ai = get_llm_with_tools().invoke(state["messages"])
    out = {"messages": [ai]}
    if not ai.tool_calls:
        out["final_reply"] = ai.content
    return out


def tools_node(state: CockpitState) -> dict:
    ai = state["messages"][-1]
    msgs, log = [], list(state.get("tool_calls_log", []))
    for call in ai.tool_calls:
        tool = TOOLS_BY_NAME.get(call["name"])
        result = (tool.invoke(call["args"]) if tool
                  else f"未知工具 {call['name']}")
        log.append((call["name"], call["args"], result))
        msgs.append(ToolMessage(result, tool_call_id=call["id"]))
    return {"messages": msgs, "tool_calls_log": log}


def route_after_agent(state: CockpitState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


def build_graph():
    b = StateGraph(CockpitState)
    b.add_node("asr", asr_node)
    b.add_node("agent", agent_node)
    b.add_node("tools", tools_node)
    b.add_edge(START, "asr")
    b.add_edge("asr", "agent")
    b.add_conditional_edges("agent", route_after_agent, ["tools", END])
    b.add_edge("tools", "agent")
    return b.compile()
