#!/usr/bin/env python
"""
Qwen3-ASR-0.6B PyTorch (bf16, CUDA) 在 AISHELL-1 test 子集上的 CER 基线。

与 Phase 1 完全同口径：同一 parquet、同 seed 抽样、同 normalize（繁→简+去标点）、
jiwer.cer。后续 ONNX/量化版本与该基线对比。

用法:
  ../venv/Scripts/python bench_cer_pytorch.py --num 200 [--seed 0]
输出:
  CER / RTF / 平均单条耗时，结果存 cer_pytorch.json
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
import soundfile as sf
import pyarrow.parquet as pq
import torch
import zhconv

HERE = Path(__file__).parent
PARQUET = HERE / "../../phase1-asr/data/aishell1_test-00000.parquet"
MODEL = "M:/models/Qwen3-ASR-0.6B"


def normalize_zh(text: str) -> str:
    """与 phase1-asr/bench/bench_wer.py 完全一致的中文 CER 规整口径。"""
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
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from qwen_asr import Qwen3ASRModel
    print("加载模型 (bf16, cuda) ...", flush=True)
    model = Qwen3ASRModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")

    print(f"加载 {args.num} 条样本 (seed={args.seed}) ...", flush=True)
    samples = load_samples(args.num, args.seed)
    dur = sum(len(a) / 16000 for a, _ in samples)
    print(f"  共 {len(samples)} 条，总时长 {dur/60:.1f} 分钟", flush=True)

    # warmup
    model.transcribe(audio=(samples[0][0], 16000))

    refs, hyps = [], []
    t_audio = t_compute = 0.0
    for i, (audio, answer) in enumerate(samples):
        t0 = time.perf_counter()
        out = model.transcribe(audio=(audio, 16000))
        t_compute += time.perf_counter() - t0
        hyp = normalize_zh(out[0].text)
        ref = normalize_zh(answer)
        if not ref:
            continue
        refs.append(ref)
        hyps.append(hyp)
        t_audio += len(audio) / 16000
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(samples)}  CER={jiwer.cer(refs, hyps)*100:.2f}%  "
                  f"RTF={t_compute/t_audio:.3f}", flush=True)

    cer = jiwer.cer(refs, hyps)
    res = {
        "model": "Qwen3-ASR-0.6B-pytorch-bf16-cuda", "n": len(refs),
        "cer": cer, "rtf": t_compute / t_audio,
        "sec_per_utt": t_compute / len(refs),
        "audio_sec": t_audio, "compute_sec": t_compute,
        "num": args.num, "seed": args.seed,
        "samples_preview": list(zip(refs[:20], hyps[:20])),
    }
    print(f"\n== Qwen3-ASR-0.6B (PyTorch bf16 CUDA): CER {cer*100:.2f}%  "
          f"RTF {res['rtf']:.3f}  {res['sec_per_utt']*1000:.0f}ms/条  (n={len(refs)})")
    with open(HERE / "cer_pytorch.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("结果已写入 cer_pytorch.json")


if __name__ == "__main__":
    main()
