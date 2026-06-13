#!/usr/bin/env python
"""
Qwen3-ASR ONNX 4-bit 权重量化（weight-only, block-wise RTN）。

INT8 已把 5 模型压到 0.91GB，但语言 decoder(443MB) + lm_head(156MB) 仍占大头。
这两块几乎全是 MatMul，正好是 ORT MatMulNBitsQuantizer 能 4-bit 的算子；其余三个
模型的权重在 Gemm(audio_transformer) / Gather(embed) / Conv(audio_frontend) 上，
NBits 量化器碰不到，且音频前端对精度敏感——所以走「混合精度」：

  decoder  + lm_head        → INT4 (MatMulNBits, block=128, HQQ 数据无关量化)
  audio_frontend/transformer/embed → 直接复用 onnx_int8 的 INT8 版本

产物 onnx_int4/ 仍是完整 5 模型目录，可被 QwenASROnnxPipeline / bench 原样加载。

⚠️ 算法选型踩坑（实测，见 bench/REPORT 第 5 节）：
  默认 RTN（round-to-nearest）4-bit 在这颗 0.6B decoder 上不稳定——简单句基本正确，
  但遇到长/难音频会进入「退化重复」生成环（输出失控变长），CER 直接爆到 60%~400%+。
  block_size 从 128 缩到 32 略有缓解但仍不可用。换成 HQQ（Half-Quadratic
  Quantization，数据无关、对 outlier 鲁棒）后 CER 基本回到 INT8 水平。所以这里用 HQQ。

用法:
  ../venv/Scripts/python quantize_int4.py --onnx ../onnx --int8 ../onnx_int8 --out ../onnx_int4
"""

import argparse
import shutil
import time
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    HQQWeightOnlyQuantConfig,
)

INT4_MODELS = ["decoder", "lm_head"]                 # MatMul 为主，4-bit
INT8_REUSE = ["audio_frontend", "audio_transformer", "embed"]  # 复用 int8


def quant_int4(src: Path, dst: Path, block_size: int):
    m = onnx.load(str(src))
    cfg = HQQWeightOnlyQuantConfig(block_size=block_size, bits=4)  # 数据无关，抗 outlier
    q = MatMulNBitsQuantizer(m, algo_config=cfg)
    q.process()
    model = q.model.model if hasattr(q.model, "model") else q.model
    onnx.save(
        model, str(dst),
        save_as_external_data=True, all_tensors_to_one_file=True,
        location=f"{dst.name}.data",
    )


def dir_size_mb(path: Path, stem: str) -> float:
    return sum(f.stat().st_size for f in path.glob(f"{stem}.onnx*")) / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="../onnx")
    ap.add_argument("--int8", default="../onnx_int8")
    ap.add_argument("--out", default="../onnx_int4")
    ap.add_argument("--block-size", type=int, default=128)
    args = ap.parse_args()

    src, int8, dst = Path(args.onnx), Path(args.int8), Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    print("== 4-bit 量化 (MatMulNBits, HQQ, block={}) ==".format(args.block_size),
          flush=True)
    total = 0.0
    for name in INT4_MODELS:
        t0 = time.perf_counter()
        quant_int4(src / f"{name}.onnx", dst / f"{name}.onnx", args.block_size)
        fp32 = (src / f"{name}.onnx").stat().st_size / 2**20
        i8 = dir_size_mb(int8, name)
        i4 = dir_size_mb(dst, name)
        total += i4
        print(f"  {name:18s} fp32 {fp32:7.1f} / int8 {i8:7.1f} / int4 {i4:7.1f} MB"
              f"  [{time.perf_counter()-t0:.0f}s]", flush=True)

    print("== 复用 INT8（Gemm/Gather/Conv，NBits 不支持）==", flush=True)
    for name in INT8_REUSE:
        for f in int8.glob(f"{name}.onnx*"):
            shutil.copy2(f, dst / f.name)
        i8 = dir_size_mb(int8, name)
        total += i8
        print(f"  {name:18s} int8 {i8:7.1f} MB (copy)", flush=True)

    print(f"\n  onnx_int4 合计 {total:7.1f} MB ({total/1024:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
