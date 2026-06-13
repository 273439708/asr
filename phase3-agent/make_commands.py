#!/usr/bin/env python
"""用 edge-tts 合成中文车控指令 wav，喂给 Qwen3-ASR 做端到端演示。

为什么自造音频：Phase 1/2 用的 AISHELL-1 是新闻朗读语料，没有车控指令句。这里用
微软在线 TTS 合成 8 条覆盖车控/导航/媒体三域（含一条多意图、一条无关指令）的指令，
统一重采样到 16kHz 单声道 wav（Qwen3-ASR 输入要求）。

诚实说明：这验证的是「端侧 ASR → Agent 工具调用」链路打通，不是 ASR 在真实车内
口音/噪声下的精度——音频是 TTS 合成的清晰普通话。

用法: ../phase2-qwen-asr/venv/Scripts/python make_commands.py
依赖: pip install edge-tts soundfile librosa
"""

import asyncio
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "commands"
VOICE = "zh-CN-XiaoxiaoNeural"

COMMANDS = [
    ("01_climate", "把空调调到二十六度"),
    ("02_window", "打开主驾驶车窗"),
    ("03_seat", "副驾座椅加热开到三档"),
    ("04_nav", "导航到上海虹桥火车站"),
    ("05_poi", "找一下附近的充电站"),
    ("06_media", "播放周杰伦的稻香"),
    ("07_multi", "有点热，把空调降到二十二度，再把天窗打开一半"),
    ("08_chat", "今天天气怎么样"),
]


async def synth_one(text: str, mp3_path: Path):
    import edge_tts
    await edge_tts.Communicate(text, VOICE).save(str(mp3_path))


def to_wav16k(mp3_path: Path, wav_path: Path):
    import librosa
    import soundfile as sf
    audio, _ = librosa.load(str(mp3_path), sr=16000, mono=True)
    sf.write(str(wav_path), audio, 16000, subtype="PCM_16")


async def main():
    OUT.mkdir(exist_ok=True)
    for name, text in COMMANDS:
        mp3, wav = OUT / f"{name}.mp3", OUT / f"{name}.wav"
        await synth_one(text, mp3)
        to_wav16k(mp3, wav)
        mp3.unlink()
        print(f"  {wav.name:16s} ← 「{text}」")
    print(f"\n共 {len(COMMANDS)} 条指令音频写入 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
