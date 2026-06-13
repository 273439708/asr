#!/usr/bin/env python
"""
运行时基准（不含 WER）：fp32 vs int8 / 无 cache vs 带 KV cache。

指标：
  - encoder 延迟（mel → hidden，30s 窗口固定 3000 帧）
  - 首 token 延迟（encoder + cross_kv 预计算 + prompt prefill + 第 1 步解码）
  - 单步解码延迟（增量 1 token）
  - RTF = 总耗时 / 音频时长（默认 30s 合成音频；--wav 可指定真实音频）

用法:
  python bench_runtime.py --onnx ../onnx --int8 ../onnx_int8 [--wav xx.wav] [--steps 64] [--runs 3]
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import onnxruntime as ort

PROMPT = [50258, 50260, 50359, 50363]  # <|sot|><|zh|><|transcribe|><|notimestamps|>
EOT = 50257
LAYERS, HEADS, HEAD_DIM = 6, 8, 64
SAMPLE_RATE = 16000


def make_session(path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def load_mel(wav_path: str | None):
    """返回 (mel[1,80,3000], 音频时长秒)。无 wav 时用 30s 合成噪声。"""
    from transformers import WhisperFeatureExtractor
    # HF 离线环境缓存中无 preprocessor_config.json，用项目内置的标准配置
    fe_dir = Path(__file__).parent / "whisper_fe"
    fe = WhisperFeatureExtractor.from_pretrained(str(fe_dir))
    if wav_path:
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        dur = len(audio) / SAMPLE_RATE
    else:
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(SAMPLE_RATE * 30) * 0.01).astype(np.float32)
        dur = 30.0
    mel = fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="np").input_features
    return mel.astype(np.float32), dur


class Pipeline:
    def __init__(self, model_dir: Path):
        self.enc = make_session(model_dir / "encoder.onnx")
        self.dec_nc = make_session(model_dir / "decoder_no_cache.onnx")
        self.cross = make_session(model_dir / "cross_kv_init.onnx")
        self.dec_c = make_session(model_dir / "decoder_with_cache.onnx")

    def encode(self, mel):
        return self.enc.run(None, {"mel": mel})[0]

    def decode_cached(self, enc_out, steps):
        """返回 (tokens, t_first_token, t_total, n_steps)。t 从 cross_kv 开始计。"""
        t0 = time.perf_counter()
        ck, cv = self.cross.run(None, {"encoder_hidden": enc_out})
        k = np.zeros((LAYERS, 1, HEADS, 0, HEAD_DIM), dtype=np.float32)
        v = np.zeros_like(k)
        logits = None
        for tok in PROMPT:
            logits, k, v = self.dec_c.run(None, {
                "token": np.array([[tok]], dtype=np.int64),
                "self_k": k, "self_v": v, "cross_k": ck, "cross_v": cv})
        seq = list(PROMPT)
        nxt = int(logits[0, -1].argmax())
        t_first = time.perf_counter() - t0
        n = 0
        while n < steps:
            seq.append(nxt)
            n += 1
            if nxt == EOT:
                break
            logits, k, v = self.dec_c.run(None, {
                "token": np.array([[nxt]], dtype=np.int64),
                "self_k": k, "self_v": v, "cross_k": ck, "cross_v": cv})
            nxt = int(logits[0, -1].argmax())
        return seq, t_first, time.perf_counter() - t0, n

    def decode_no_cache(self, enc_out, steps):
        t0 = time.perf_counter()
        seq = list(PROMPT)
        t_first = None
        n = 0
        while n < steps:
            logits = self.dec_nc.run(None, {
                "tokens": np.array([seq], dtype=np.int64),
                "encoder_hidden": enc_out})[0]
            nxt = int(logits[0, -1].argmax())
            if t_first is None:
                t_first = time.perf_counter() - t0
            seq.append(nxt)
            n += 1
            if nxt == EOT:
                break
        return seq, t_first, time.perf_counter() - t0, n


def bench(tag: str, pipe: Pipeline, mel, dur: float, steps: int, runs: int):
    print(f"\n== {tag} ==")
    # encoder（warmup 1 次后取 runs 次最优）
    enc_out = pipe.encode(mel)
    t_enc = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        enc_out = pipe.encode(mel)
        t_enc = min(t_enc, time.perf_counter() - t0)

    rows = []
    for name, fn in [("with_cache", pipe.decode_cached),
                     ("no_cache", pipe.decode_no_cache)]:
        best = None
        for _ in range(runs):
            seq, t_first, t_dec, n = fn(enc_out, steps)
            if best is None or t_dec < best[2]:
                best = (seq, t_first, t_dec, n)
        seq, t_first, t_dec, n = best
        total = t_enc + t_dec
        rows.append((name, t_first, t_dec / max(n, 1), total, total / dur, n))

    print(f"  encoder            : {t_enc*1000:7.1f} ms")
    print(f"  {'decode':18s} {'首token(ms)':>12s} {'ms/step':>9s} {'端到端(s)':>10s} {'RTF':>7s} {'steps':>6s}")
    for name, t_first, per_step, total, rtf, n in rows:
        t1 = (t_enc + t_first) * 1000  # 首 token 含 encoder
        print(f"  {name:18s} {t1:12.1f} {per_step*1000:9.1f} {total:10.3f} {rtf:7.3f} {n:6d}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="../onnx")
    ap.add_argument("--int8", default="../onnx_int8")
    ap.add_argument("--wav", default=None)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    mel, dur = load_mel(args.wav)
    src = "合成噪声" if args.wav is None else args.wav
    print(f"音频: {src}  时长 {dur:.1f}s  mel={mel.shape}")
    if args.wav is None:
        print("注意: 合成音频上解码内容无意义，RTF 以固定 steps 截断计；"
              "真实 RTF 请用 --wav 指定语音文件。")

    for tag, d in [("fp32", args.onnx), ("int8", args.int8)]:
        bench(tag, Pipeline(Path(d)), mel, dur, args.steps, args.runs)


if __name__ == "__main__":
    main()
