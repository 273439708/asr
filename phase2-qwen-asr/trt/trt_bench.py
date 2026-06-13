#!/usr/bin/env python
"""
Qwen3-ASR 音频编码器 TensorRT FP16 vs ONNX Runtime (CPU/CUDA EP) 延迟对比。

TRT 11 Python API（pip wheel tensorrt-cu12，无 trtexec）。TRT 11 移除全部弱类型
精度 flag（BuilderFlag.FP16 等），网络默认强类型、精度跟随模型 dtype，因此先用
onnxconverter-common 把 fp32 ONNX 转 fp16（keep_io_types=True，I/O 保持 fp32）：
  audio_frontend.onnx   → *_fp16.onnx → frontend_fp16.plan   （动态轴 N：1/8/64）
  audio_transformer.onnx→ *_fp16.onnx → transformer_fp16.plan（动态轴 S：13/104/832，
                                                               attn_mask 同步 profile）
engine 构建 ~30s，缓存到 --plan 目录，复跑直接反序列化。

精度：与 ORT CPU fp32 输出比 max/mean diff + 余弦（FP16 预期 ~1e-2 量级 diff、cos>0.998）。
延迟：同一真实样本重复 N 次取均值（含 H2D/D2H 拷贝）。

用法:
  ../venv/Scripts/python trt_bench.py --onnx ../onnx --plan ../trt_plans [--iters 50]
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "../export"))
from onnx_pipeline import QwenASROnnxPipeline, CHUNK, CHUNK_OUT, WINDOW, conv_out_len  # noqa: E402

PARQUET = HERE / "../../phase1-asr/data/aishell1_test-00000.parquet"
MODEL = "M:/models/Qwen3-ASR-0.6B"
NEG_FP16 = -1e4  # fp16 安全的加性 mask 值（fp32 min 在 fp16 下溢出为 -inf）


# ------------------------------------------------------------------ TRT build

def to_fp16_onnx(src: Path, dst: Path) -> Path:
    """fp32 ONNX → fp16 ONNX（keep_io_types：I/O 保持 fp32，内部插 Cast）。
    TRT 11 移除了弱类型 BuilderFlag.FP16，网络默认强类型，精度跟随模型 dtype。"""
    if dst.exists():
        print(f"  复用 {dst.name}")
        return dst
    import onnx
    from onnxconverter_common import float16
    print(f"  转 fp16: {src.name} ...", flush=True)
    m = onnx.load(str(src))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m16, str(dst))
    print(f"  {dst.name} {dst.stat().st_size/2**20:.1f} MB")
    return dst


def build_engine(onnx_path: Path, plan_path: Path, profiles: dict, fp16: bool = True):
    """profiles: {input_name: (min_shape, opt_shape, max_shape)}"""
    import tensorrt as trt
    if plan_path.exists():
        print(f"  复用缓存 {plan_path.name}")
        return plan_path.read_bytes()
    if fp16:
        onnx_path = to_fp16_onnx(
            onnx_path, plan_path.parent / onnx_path.name.replace(".onnx", "_fp16.onnx"))
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TRT 11: 强类型是默认且唯一模式
    parser = trt.OnnxParser(network, logger)
    print(f"  解析 {onnx_path.name} ...", flush=True)
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print("   ", parser.get_error(i))
        raise RuntimeError(f"parse failed: {onnx_path}")
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    for name, (mn, opt, mx) in profiles.items():
        profile.set_shape(name, mn, opt, mx)
    config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    print(f"  构建 engine (FP16={fp16}) ...", flush=True)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"build failed: {onnx_path}")
    blob = bytes(blob)  # IHostMemory → bytes（TRT 11 IHostMemory 无 len()）
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(blob)
    print(f"  {plan_path.name} {len(blob)/2**20:.1f} MB  [{time.perf_counter()-t0:.0f}s]")
    return blob


class TrtModule:
    """单 profile 动态形状 engine 的最小执行封装（torch cuda tensor 做 I/O 缓冲）。"""

    def __init__(self, blob: bytes):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def __call__(self, feeds: dict) -> torch.Tensor:
        """feeds: {name: torch cuda tensor}；返回第一个输出（cuda tensor）。"""
        e, ctx = self.engine, self.ctx
        out_name = None
        for i in range(e.num_io_tensors):
            name = e.get_tensor_name(i)
            if e.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                ctx.set_input_shape(name, tuple(feeds[name].shape))
                ctx.set_tensor_address(name, feeds[name].data_ptr())
            else:
                out_name = name
        out = torch.empty(tuple(ctx.get_tensor_shape(out_name)),
                          dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(out_name, out.data_ptr())
        ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out


class TrtAudioEncoder:
    """与 QwenASROnnxPipeline.encode_audio 同逻辑的 TRT 版本。"""

    def __init__(self, frontend: TrtModule, transformer: TrtModule):
        self.frontend = frontend
        self.transformer = transformer

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        T = mel.shape[1]
        n_full, tail = T // CHUNK, T % CHUNK
        n = n_full + (1 if tail else 0)
        padded = np.zeros((n * CHUNK, 128), dtype=np.float32)
        padded[:T] = mel.T
        chunks = torch.from_numpy(
            padded.reshape(n, CHUNK, 128).transpose(0, 2, 1)[:, None].copy()).cuda()
        chunk_embeds = self.frontend({"chunks": chunks})            # [N,13,896]

        valid = [CHUNK_OUT] * n_full + ([conv_out_len(tail)] if tail else [])
        hidden = torch.cat([chunk_embeds[i, :v] for i, v in enumerate(valid)])  # [S,896]
        S = hidden.shape[0]
        mask = torch.full((1, 1, S, S), NEG_FP16, dtype=torch.float32, device="cuda")
        start = 0
        while start < S:
            end = min(start + WINDOW, S)
            mask[..., start:end, start:end] = 0.0
            start = end
        out = self.transformer({"hidden": hidden.contiguous(), "attn_mask": mask})
        return out.cpu().numpy()


# ----------------------------------------------------------------------- main

def load_sample(seconds_pad: float = 0.0):
    table = pq.ParquetFile(PARQUET).read(columns=["context"])
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(table.num_rows, size=3, replace=False))
    rows = table.take(idx).to_pylist()
    audios = [sf.read(io.BytesIO(r["context"]["bytes"]), dtype="float32")[0] for r in rows]
    if seconds_pad:  # 拼接成长音频，看长序列下的差距
        audios.append(np.concatenate(audios * int(seconds_pad)))
    return audios


def make_mels(audios):
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(MODEL)
    return [fe(a, sampling_rate=16000, return_attention_mask=True,
               padding="do_not_pad", return_tensors="np").input_features[0].astype(np.float32)
            for a in audios]


def bench(fn, arg, iters):
    fn(arg)  # warmup
    fn(arg)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(arg)
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="../onnx")
    ap.add_argument("--plan", default="../trt_plans")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    onnx_dir, plan_dir = Path(args.onnx), Path(args.plan)

    import tensorrt as trt
    print(f"TensorRT {trt.__version__}  GPU {torch.cuda.get_device_name(0)}")

    # 1. 构建/加载 engine
    print("\n== TRT FP16 engine ==")
    blob_fe = build_engine(
        onnx_dir / "audio_frontend.onnx", plan_dir / "frontend_fp16.plan",
        {"chunks": ((1, 1, 128, 100), (8, 1, 128, 100), (64, 1, 128, 100))})
    blob_tr = build_engine(
        onnx_dir / "audio_transformer.onnx", plan_dir / "transformer_fp16.plan",
        {"hidden": ((13, 896), (104, 896), (832, 896)),
         "attn_mask": ((1, 1, 13, 13), (1, 1, 104, 104), (1, 1, 832, 832))})
    trt_enc = TrtAudioEncoder(TrtModule(blob_fe), TrtModule(blob_tr))

    # 2. ORT 对照（CPU EP 与 CUDA EP）
    import onnxruntime as ort
    print(f"\nonnxruntime {ort.__version__}  providers={ort.get_available_providers()}")
    pipe_cpu = QwenASROnnxPipeline(onnx_dir)

    cuda_ok = "CUDAExecutionProvider" in ort.get_available_providers()
    pipe_cuda = None
    if cuda_ok:
        try:
            pipe_cuda = QwenASROnnxPipeline(onnx_dir, providers=[
                "CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception as e:
            print(f"  CUDA EP 不可用，跳过: {e}")

    # 3. 精度 + 延迟
    audios = load_sample(seconds_pad=4)   # 3 条短句 + 1 条 ~40s 拼接长音频
    mels = make_mels(audios)
    print(f"\n== 音频编码器延迟 (ms, {args.iters} 次均值) 与 TRT-FP16 精度 ==")
    print(f"{'mel T':>6} {'S':>5} | {'ORT CPU fp32':>12} {'ORT CUDA fp32':>13} "
          f"{'TRT FP16':>9} | {'max diff':>9} {'cos':>8}")
    for mel in mels:
        T = mel.shape[1]
        ref = pipe_cpu.encode_audio(mel)
        out = trt_enc(mel)
        d = np.abs(ref - out)
        cos = float((ref * out).sum() / (np.linalg.norm(ref) * np.linalg.norm(out)))
        t_cpu = bench(pipe_cpu.encode_audio, mel, max(args.iters // 5, 5))
        t_cuda = bench(pipe_cuda.encode_audio, mel, args.iters) if pipe_cuda else float("nan")
        t_trt = bench(trt_enc, mel, args.iters)
        print(f"{T:>6} {ref.shape[0]:>5} | {t_cpu:>12.1f} {t_cuda:>13.1f} "
              f"{t_trt:>9.1f} | {d.max():>9.2e} {cos:>8.5f}")

    print("\n注: TRT/CUDA 计时含 H2D/D2H 拷贝；mask 用 -1e4（fp32 min 在 fp16 溢出为 -inf）。")


if __name__ == "__main__":
    main()
