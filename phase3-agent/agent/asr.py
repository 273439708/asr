"""ASR 节点：薄封装 Phase 2 的 QwenASROnnxPipeline。
音频 wav → mel（WhisperFeatureExtractor）→ ONNX int8 转写 → 取 <asr_text> 后的文本。"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "../../phase2-qwen-asr/export"))
from onnx_pipeline import QwenASROnnxPipeline  # noqa: E402

MODEL = "M:/models/Qwen3-ASR-0.6B"
# 默认用 Phase 2 的 int8 产物（端侧主推），缺失时回落 fp32
ONNX_INT8 = HERE / "../../phase2-qwen-asr/onnx_int8"
ONNX_FP32 = HERE / "../../phase2-qwen-asr/onnx"
ASR_TEXT = 151704  # <asr_text>：其后为转写文本


class CockpitASR:
    def __init__(self, onnx_dir: Path | None = None):
        if onnx_dir is None:
            onnx_dir = ONNX_INT8 if (ONNX_INT8 / "decoder.onnx").exists() else ONNX_FP32
        from transformers import AutoTokenizer, WhisperFeatureExtractor
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.fe = WhisperFeatureExtractor.from_pretrained(MODEL)
        self.pipe = QwenASROnnxPipeline(str(onnx_dir))
        self.onnx_dir = Path(onnx_dir)

    def transcribe(self, wav_path: str) -> tuple[str, float]:
        """返回 (识别文本, 总耗时秒)。"""
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:           # 立体声取单声道
            audio = audio.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"需要 16kHz 音频，得到 {sr}Hz：{wav_path}")
        mel = self.fe(audio, sampling_rate=16000, return_attention_mask=True,
                      padding="do_not_pad", return_tensors="np"
                      ).input_features[0].astype(np.float32)
        toks, _, t_total = self.pipe.transcribe(mel)
        body = toks[toks.index(ASR_TEXT) + 1:] if ASR_TEXT in toks else toks
        text = self.tok.decode(body, skip_special_tokens=True).strip()
        return text, t_total
