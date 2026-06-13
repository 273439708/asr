#!/usr/bin/env python
"""
ONNX 导出验证：
  1. 数值对齐：encoder / decoder(no cache) / cross_kv / decoder(with cache)
     ONNX Runtime 输出 vs PyTorch 输出的最大绝对误差
  2. 解码一致性 + 速度：同一段 mel 上贪心解码 N 步，
     无 cache（每步重算全序列） vs 带 cache（每步 1 token），
     验证 token 序列完全一致，并对比耗时。

用法:
  python verify_onnx.py --model openai/whisper-base --onnx ../onnx --steps 64
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import WhisperForConditionalGeneration

from export_whisper_onnx import (
    CrossKVInit,
    DecoderNoCache,
    DecoderWithCache,
    EncoderWrapper,
    patch_mask_for_onnx_export,
)


def max_diff(a: torch.Tensor, b: np.ndarray) -> float:
    return float(np.abs(a.detach().numpy() - b).max())


def make_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--onnx", default="../onnx")
    ap.add_argument("--steps", type=int, default=64, help="贪心解码步数")
    args = ap.parse_args()

    onnx_dir = Path(args.onnx)
    patch_mask_for_onnx_export()

    print(f"loading {args.model} ...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.eval()
    cfg = model.config
    layers, heads = cfg.decoder_layers, cfg.decoder_attention_heads
    head_dim = cfg.d_model // heads

    enc_sess = make_session(onnx_dir / "encoder.onnx")
    dec_nc_sess = make_session(onnx_dir / "decoder_no_cache.onnx")
    cross_sess = make_session(onnx_dir / "cross_kv_init.onnx")
    dec_c_sess = make_session(onnx_dir / "decoder_with_cache.onnx")

    torch.manual_seed(0)
    mel = torch.randn(1, cfg.num_mel_bins, 3000)

    # ---------- 1. 数值对齐 ----------
    print("\n== 数值对齐 (ONNX vs PyTorch, max abs diff) ==")
    with torch.no_grad():
        enc_pt = EncoderWrapper(model)(mel)
    enc_ox = enc_sess.run(None, {"mel": mel.numpy()})[0]
    print(f"encoder            : {max_diff(enc_pt, enc_ox):.3e}")

    tokens = torch.tensor([[50258, 50260, 50359, 50363]], dtype=torch.long)
    with torch.no_grad():
        logits_pt = DecoderNoCache(model)(tokens, enc_pt)
    logits_ox = dec_nc_sess.run(
        None, {"tokens": tokens.numpy(), "encoder_hidden": enc_ox})[0]
    print(f"decoder_no_cache   : {max_diff(logits_pt, logits_ox):.3e}")

    with torch.no_grad():
        ck_pt, cv_pt = CrossKVInit(model)(enc_pt)
    ck_ox, cv_ox = cross_sess.run(None, {"encoder_hidden": enc_ox})
    print(f"cross_kv_init (k)  : {max_diff(ck_pt, ck_ox):.3e}")
    print(f"cross_kv_init (v)  : {max_diff(cv_pt, cv_ox):.3e}")

    past = 4
    token1 = torch.tensor([[50363]], dtype=torch.long)
    sk = torch.randn(layers, 1, heads, past, head_dim)
    sv = torch.randn(layers, 1, heads, past, head_dim)
    with torch.no_grad():
        lg_pt, nk_pt, nv_pt = DecoderWithCache(model)(token1, sk, sv, ck_pt, cv_pt)
    lg_ox, nk_ox, nv_ox = dec_c_sess.run(None, {
        "token": token1.numpy(), "self_k": sk.numpy(), "self_v": sv.numpy(),
        "cross_k": ck_ox, "cross_v": cv_ox})
    print(f"decoder_with_cache : logits {max_diff(lg_pt, lg_ox):.3e}"
          f"  new_k {max_diff(nk_pt, nk_ox):.3e}  new_v {max_diff(nv_pt, nv_ox):.3e}")

    # ---------- 2. 贪心解码：无 cache vs 带 cache ----------
    print(f"\n== 贪心解码 {args.steps} 步 (ONNX Runtime CPU) ==")
    init = [50258, 50260, 50359, 50363]  # <|sot|><|zh|><|transcribe|><|notimestamps|>

    # 2a. 无 cache：每步把全部历史 token 重新喂入
    t0 = time.perf_counter()
    seq_nc = list(init)
    for _ in range(args.steps):
        out = dec_nc_sess.run(None, {
            "tokens": np.array([seq_nc], dtype=np.int64),
            "encoder_hidden": enc_ox})[0]
        seq_nc.append(int(out[0, -1].argmax()))
    t_nc = time.perf_counter() - t0

    # 2b. 带 cache：先预填充 init tokens，再逐 token 增量
    t0 = time.perf_counter()
    k = np.zeros((layers, 1, heads, 0, head_dim), dtype=np.float32)
    v = np.zeros((layers, 1, heads, 0, head_dim), dtype=np.float32)
    seq_c = list(init)
    last_logits = None
    for tok in init:  # prefill（逐 token，与增量路径同一模型）
        last_logits, k, v = dec_c_sess.run(None, {
            "token": np.array([[tok]], dtype=np.int64),
            "self_k": k, "self_v": v, "cross_k": ck_ox, "cross_v": cv_ox})
    for _ in range(args.steps):
        nxt = int(last_logits[0, -1].argmax())
        seq_c.append(nxt)
        last_logits, k, v = dec_c_sess.run(None, {
            "token": np.array([[nxt]], dtype=np.int64),
            "self_k": k, "self_v": v, "cross_k": ck_ox, "cross_v": cv_ox})
    t_c = time.perf_counter() - t0

    same = seq_nc == seq_c
    print(f"token 序列一致     : {same}")
    if not same:
        print(f"  no_cache  : {seq_nc}")
        print(f"  with_cache: {seq_c}")
    print(f"no KV cache        : {t_nc:.3f}s  ({t_nc/args.steps*1000:.1f} ms/step)")
    print(f"with KV cache      : {t_c:.3f}s  ({t_c/args.steps*1000:.1f} ms/step)")
    print(f"加速比             : {t_nc/t_c:.2f}x")


if __name__ == "__main__":
    main()
