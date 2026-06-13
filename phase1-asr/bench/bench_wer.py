#!/usr/bin/env python
"""
AISHELL-1 test 子集上评测 fp32 / int8 ONNX 管线的 CER（中文按字算，习惯仍称 WER）。

流程：parquet 读音频 + 参考文本 → WhisperFeatureExtractor → encoder →
     cross_kv_init → decoder_with_cache 贪心解码 → tokenizer 解码 →
     文本规整（去标点/空格）→ jiwer.cer

用法:
  python bench_wer.py --models ../onnx ../onnx_int8 --num 200 [--seed 0]
输出:
  每套模型的 CER / RTF / 平均首 token 延迟，结果存 wer_results.json
"""

import argparse
import io
import json
import re
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
import zhconv
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

HERE = Path(__file__).parent
PARQUET = HERE / "../data/aishell1_test-00000.parquet"
PROMPT = [50258, 50260, 50359, 50363]  # <|sot|><|zh|><|transcribe|><|notimestamps|>
EOT = 50257
LAYERS, HEADS, HEAD_DIM = 6, 8, 64
MAX_TOKENS = 128


def make_session(path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def normalize_zh(text: str) -> str:
    """繁→简 + 去标点/空格/全角符号，仅保留汉字与字母数字，中文 CER 口径。
    whisper-base 对 zh 常输出繁体（训练数据混合），不转换会被 CER 全判错。"""
    text = zhconv.convert(text, "zh-cn")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w]", "", text)  # \w 含汉字
    return text.lower()


class Pipeline:
    def __init__(self, model_dir: Path):
        self.enc = make_session(model_dir / "encoder.onnx")
        self.cross = make_session(model_dir / "cross_kv_init.onnx")
        self.dec = make_session(model_dir / "decoder_with_cache.onnx")

    def transcribe(self, mel):
        """返回 (token list 不含特殊符, 首token延迟, 总耗时)"""
        t0 = time.perf_counter()
        enc_out = self.enc.run(None, {"mel": mel})[0]
        ck, cv = self.cross.run(None, {"encoder_hidden": enc_out})
        k = np.zeros((LAYERS, 1, HEADS, 0, HEAD_DIM), dtype=np.float32)
        v = np.zeros_like(k)
        logits = None
        for tok in PROMPT:
            logits, k, v = self.dec.run(None, {
                "token": np.array([[tok]], dtype=np.int64),
                "self_k": k, "self_v": v, "cross_k": ck, "cross_v": cv})
        nxt = int(logits[0, -1].argmax())
        t_first = time.perf_counter() - t0
        out = []
        while nxt != EOT and len(out) < MAX_TOKENS:
            out.append(nxt)
            logits, k, v = self.dec.run(None, {
                "token": np.array([[nxt]], dtype=np.int64),
                "self_k": k, "self_v": v, "cross_k": ck, "cross_v": cv})
            nxt = int(logits[0, -1].argmax())
        return out, t_first, time.perf_counter() - t0


def load_samples(num: int, seed: int):
    pf = pq.ParquetFile(PARQUET)
    table = pf.read(columns=["context", "answer"])
    n_total = table.num_rows
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(n_total, size=min(num, n_total), replace=False))
    rows = table.take(idx).to_pylist()
    samples = []
    for r in rows:
        audio, sr = sf.read(io.BytesIO(r["context"]["bytes"]), dtype="float32")
        assert sr == 16000
        samples.append((audio, r["answer"]))
    return samples


def evaluate(tag: str, model_dir: Path, samples, fe, tok):
    pipe = Pipeline(model_dir)
    refs, hyps = [], []
    t_audio = t_compute = 0.0
    t_firsts = []
    for i, (audio, answer) in enumerate(samples):
        mel = fe(audio, sampling_rate=16000,
                 return_tensors="np").input_features.astype(np.float32)
        ids, t_first, t_dec = pipe.transcribe(mel)
        hyp = normalize_zh(tok.decode(ids, skip_special_tokens=True))
        ref = normalize_zh(answer)
        if not ref:
            continue
        refs.append(ref)
        hyps.append(hyp)
        t_audio += len(audio) / 16000
        t_compute += t_dec
        t_firsts.append(t_first)
        if (i + 1) % 25 == 0:
            running = jiwer.cer(refs, hyps)
            print(f"  [{tag}] {i+1}/{len(samples)}  CER={running*100:.2f}%  "
                  f"RTF={t_compute/t_audio:.3f}", flush=True)
    cer = jiwer.cer(refs, hyps)
    res = {
        "model": tag, "n": len(refs),
        "cer": cer, "rtf": t_compute / t_audio,
        "first_token_ms": float(np.mean(t_firsts) * 1000),
        "audio_sec": t_audio, "compute_sec": t_compute,
    }
    print(f"== {tag}: CER {cer*100:.2f}%  RTF {res['rtf']:.3f}  "
          f"首token {res['first_token_ms']:.0f}ms  (n={len(refs)})")
    return res, list(zip(refs, hyps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["../onnx", "../onnx_int8"])
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fe = WhisperFeatureExtractor.from_pretrained(str(HERE / "whisper_fe"))
    tok = WhisperTokenizerFast.from_pretrained(str(HERE / "whisper_tok"))

    print(f"加载 {args.num} 条样本 (seed={args.seed}) ...", flush=True)
    samples = load_samples(args.num, args.seed)
    dur = sum(len(a) / 16000 for a, _ in samples)
    print(f"  共 {len(samples)} 条，总时长 {dur/60:.1f} 分钟", flush=True)

    results, transcripts = [], {}
    for d in args.models:
        d = Path(d)
        tag = d.name
        res, pairs = evaluate(tag, d, samples, fe, tok)
        results.append(res)
        transcripts[tag] = pairs[:20]  # 留 20 条供抽查

    out = {"num": args.num, "seed": args.seed, "results": results,
           "samples_preview": transcripts}
    with open(HERE / "wer_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果已写入 wer_results.json")


if __name__ == "__main__":
    main()
