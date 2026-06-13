#!/usr/bin/env python
"""
从 AISHELL-1 test parquet 抽取固定测试子集：
  testset/wav/0000.wav ... + testset/transcripts.tsv (id \t text)

用法:
  python extract_testset.py --parquet aishell1_test_00.parquet --n 200
"""

import argparse
import wave
import io
from pathlib import Path

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="aishell1_test_00.parquet")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="testset")
    args = ap.parse_args()

    out = Path(args.out)
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    f = pq.ParquetFile(args.parquet)
    rows = []
    total_dur = 0.0
    it = f.iter_batches(batch_size=64)
    idx = 0
    for batch in it:
        for r in batch.to_pylist():
            if idx >= args.n:
                break
            data = r["context"]["bytes"]
            with wave.open(io.BytesIO(data)) as w:
                sr = w.getframerate()
                dur = w.getnframes() / sr
            assert sr == 16000, f"unexpected sr={sr}"
            name = f"{idx:04d}"
            (wav_dir / f"{name}.wav").write_bytes(data)
            rows.append((name, r["answer"].strip()))
            total_dur += dur
            idx += 1
        if idx >= args.n:
            break

    with open(out / "transcripts.tsv", "w", encoding="utf-8") as fp:
        for name, text in rows:
            fp.write(f"{name}\t{text}\n")

    print(f"{idx} utts, total {total_dur:.1f}s audio ({total_dur/idx:.2f}s avg)")
    print(f"-> {wav_dir}  +  {out / 'transcripts.tsv'}")


if __name__ == "__main__":
    main()
