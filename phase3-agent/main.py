#!/usr/bin/env python
"""智能座舱语音助手端到端演示：车控指令音频 → Qwen3-ASR → LangGraph Agent → 模拟车控。

对每条 commands/*.wav：跑图（asr→agent→tools→agent 闭环），打印
  识别文本 / 工具调用轨迹 / 车辆状态增量 / 助手回复。

用法:
  ../phase2-qwen-asr/venv/Scripts/python main.py            # 跑全部 commands/*.wav
  ../phase2-qwen-asr/venv/Scripts/python main.py 07_multi   # 只跑指定指令
依赖 .env（复制 .env.example，填 deepseek 密钥）。
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，强制 UTF-8 输出

from agent.graph import build_graph
from agent.tools import reset_vehicle, vehicle_snapshot

HERE = Path(__file__).parent
CMD_DIR = HERE / "commands"


def diff_state(before: dict, after: dict) -> list[str]:
    """递归对比两份车辆状态，返回变化项的人类可读描述。"""
    changes = []
    for k, av in after.items():
        bv = before.get(k)
        if isinstance(av, dict):
            for sk, sav in av.items():
                if bv.get(sk) != sav:
                    changes.append(f"{k}.{sk}: {bv.get(sk)} → {sav}")
        elif bv != av:
            changes.append(f"{k}: {bv} → {av}")
    return changes


def run_one(graph, wav: Path):
    reset_vehicle()
    before = vehicle_snapshot()
    print(f"\n{'='*64}\n♪ 音频: {wav.name}")
    state = graph.invoke({"audio_path": str(wav)})
    print(f"  识别文本: {state['transcript']}")
    log = state.get("tool_calls_log", [])
    if log:
        print("  工具调用:")
        for name, args, result in log:
            print(f"    → {name}({args}) = {result}")
    else:
        print("  工具调用: (无)")
    changes = diff_state(before, vehicle_snapshot())
    if changes:
        print("  车辆状态变化: " + "; ".join(changes))
    print(f"  助手回复: {state.get('final_reply', '(无)')}")


def main():
    if not CMD_DIR.exists() or not list(CMD_DIR.glob("*.wav")):
        sys.exit(f"未找到指令音频，请先运行 make_commands.py 生成 {CMD_DIR}/*.wav")
    graph = build_graph()
    if len(sys.argv) > 1:
        wavs = [CMD_DIR / f"{sys.argv[1].removesuffix('.wav')}.wav"]
    else:
        wavs = sorted(CMD_DIR.glob("*.wav"))
    for wav in wavs:
        run_one(graph, wav)


if __name__ == "__main__":
    main()
