#!/usr/bin/env python
"""
Qwen3-ASR ONNX（fp32 或 int8, CPU）在 AISHELL-1 test 子集上的 CER / RTF 评测。

与 bench_cer_pytorch.py 完全同口径（同 parquet、同 seed 抽样、同 normalize、jiwer.cer），
直接与 PyTorch bf16 CUDA 基线 3.71% 对比。

用法:
  ../venv/Scripts/python bench_cer_onnx.py --onnx ../onnx --tag fp32 --num 200
  ../venv/Scripts/python bench_cer_onnx.py --onnx ../onnx_int8 --tag int8 --num 200
"""

import argparse
import io
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import zhconv

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "../export"))
from onnx_pipeline import QwenASROnnxPipeline  # noqa: E402

PARQUET = HERE / "../../phase1-asr/data/aishell1_test-00000.parquet"
MODEL = "M:/models/Qwen3-ASR-0.6B"
ASR_TEXT = 151704  # <asr_text>：之前为语种标签（language Chinese），之后为转写文本


def normalize_zh(text: str) -> str:
    text = zhconv.convert(text, "zh-cn")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w]", "", text)
    return text.lower()


def load_samples(num: int, seed: int):
    pf = pq.ParquetFile(PARQUET)
    table = pf.read(columns=["context", "answer"])
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(table.num_rows, size=min(num, table.num_rows), replace=False))
    rows = table.take(idx).to_pylist()
    samples = []
    for r in rows:
        audio, sr = sf.read(io.BytesIO(r["context"]["bytes"]), dtype="float32")
        assert sr == 16000
        samples.append((audio, r["answer"]))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="../onnx")
    ap.add_argument("--tag", default="fp32")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer, WhisperFeatureExtractor
    tok = AutoTokenizer.from_pretrained(MODEL)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL)

    print(f"加载 ONNX 管线 ({args.onnx}) ...", flush=True)
    pipe = QwenASROnnxPipeline(args.onnx)
    samples = load_samples(args.num, args.seed)
    dur = sum(len(a) / 16000 for a, _ in samples)
    print(f"  共 {len(samples)} 条，总时长 {dur/60:.1f} 分钟", flush=True)

    refs, hyps = [], []
    t_audio = t_compute = t_first_sum = 0.0
    for i, (audio, answer) in enumerate(samples):
        feats = fe(audio, sampling_rate=16000, return_attention_mask=True,
                   padding="do_not_pad", return_tensors="np")
        mel = feats.input_features[0].astype(np.float32)
        toks, t_first, t_total = pipe.transcribe(mel)
        # 取 <asr_text> 之后的转写文本
        text = tok.decode(toks[toks.index(ASR_TEXT) + 1:] if ASR_TEXT in toks else toks,
                          skip_special_tokens=True)
        hyp = normalize_zh(text)
        ref = normalize_zh(answer)
        if not ref:
            continue
        refs.append(ref)
        hyps.append(hyp)
        t_audio += len(audio) / 16000
        t_compute += t_total
        t_first_sum += t_first
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(samples)}  CER={jiwer.cer(refs, hyps)*100:.2f}%  "
                  f"RTF={t_compute/t_audio:.3f}", flush=True)

    cer = jiwer.cer(refs, hyps)
    res = {
        "model": f"Qwen3-ASR-0.6B-onnx-{args.tag}-cpu", "n": len(refs),
        "cer": cer, "rtf": t_compute / t_audio,
        "sec_per_utt": t_compute / len(refs),
        "first_token_ms": t_first_sum / len(refs) * 1000,
        "audio_sec": t_audio, "compute_sec": t_compute,
        "num": args.num, "seed": args.seed,
        "samples_preview": list(zip(refs[:20], hyps[:20])),
    }
    print(f"\n== Qwen3-ASR-0.6B (ONNX {args.tag} CPU): CER {cer*100:.2f}%  "
          f"RTF {res['rtf']:.3f}  {res['sec_per_utt']*1000:.0f}ms/条  "
          f"首token {res['first_token_ms']:.0f}ms  (n={len(refs)})")
    out = HERE / f"cer_onnx_{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {out.name}")


if __name__ == "__main__":
    main()
