#!/usr/bin/env python
"""
ONNX 动态 INT8 量化 + 量化前后精度对齐。

  1. 对 4 个导出模型做 quantize_dynamic（MatMul/Gemm 权重 int8）
  2. 固定输入下对比 fp32 / int8 最终输出误差
  3. 贪心解码 token 一致性 + 解码速度对比
  4. --layerwise: 逐层（每个 decoder layer 输出）误差定位，看误差在哪层开始放大

用法:
  python quantize_int8.py --onnx ../onnx --out ../onnx_int8 [--layerwise]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

MODELS = ["encoder", "decoder_no_cache", "cross_kv_init", "decoder_with_cache"]
PROMPT = [50258, 50260, 50359, 50363]  # <|sot|><|zh|><|transcribe|><|notimestamps|>

# whisper-base 结构参数（与导出一致）
LAYERS, HEADS, HEAD_DIM, N_MELS = 6, 8, 64, 80


def make_session(path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def quantize_all(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    print("== 动态 INT8 量化 ==")
    for name in MODELS:
        fp32 = src / f"{name}.onnx"
        int8 = dst / f"{name}.onnx"
        # 配置依据（实验对比见 git 历史 / bench 报告）：
        # - 只量化 MatMul/Gemm/Gather：ORT CPU EP 没有 ConvInteger 实现，
        #   encoder 前端 2 个 Conv 保持 fp32
        # - Gather 量化的是 token embedding 权重（decoder 体积减半），实测 token 全一致
        # - per_channel=True：encoder 输出 mean|diff| 0.128→0.076，余弦 0.9936→0.9978
        quantize_dynamic(
            str(fp32), str(int8), weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm", "Gather"],
            per_channel=True,
        )
        s0, s1 = fp32.stat().st_size / 1e6, int8.stat().st_size / 1e6
        print(f"  {name:22s} {s0:7.1f} MB -> {s1:7.1f} MB  ({s1/s0*100:.0f}%)")


def greedy_with_cache(sess_cross, sess_dec, enc, steps):
    """带 cache 贪心解码，返回 (tokens, 耗时)"""
    ck, cv = sess_cross.run(None, {"encoder_hidden": enc})
    k = np.zeros((LAYERS, 1, HEADS, 0, HEAD_DIM), dtype=np.float32)
    v = np.zeros_like(k)
    seq = list(PROMPT)
    t0 = time.perf_counter()
    logits = None
    for tok in PROMPT:
        logits, k, v = sess_dec.run(None, {
            "token": np.array([[tok]], dtype=np.int64),
            "self_k": k, "self_v": v, "cross_k": ck, "cross_v": cv})
    for _ in range(steps):
        nxt = int(logits[0, -1].argmax())
        seq.append(nxt)
        logits, k, v = sess_dec.run(None, {
            "token": np.array([[nxt]], dtype=np.int64),
            "self_k": k, "self_v": v, "cross_k": ck, "cross_v": cv})
    return seq, time.perf_counter() - t0


def check_outputs(src: Path, dst: Path, steps: int):
    print("\n== fp32 vs int8 输出对齐 (max abs diff) ==")
    rng = np.random.default_rng(0)
    mel = rng.standard_normal((1, N_MELS, 3000), dtype=np.float32)

    enc_f = make_session(src / "encoder.onnx")
    enc_q = make_session(dst / "encoder.onnx")
    e_f = enc_f.run(None, {"mel": mel})[0]
    e_q = enc_q.run(None, {"mel": mel})[0]
    print(f"  encoder            : {np.abs(e_f - e_q).max():.3e}"
          f"  (输出幅值 ~{np.abs(e_f).mean():.2f})")

    nc_f = make_session(src / "decoder_no_cache.onnx")
    nc_q = make_session(dst / "decoder_no_cache.onnx")
    toks = np.array([PROMPT], dtype=np.int64)
    l_f = nc_f.run(None, {"tokens": toks, "encoder_hidden": e_f})[0]
    l_q = nc_q.run(None, {"tokens": toks, "encoder_hidden": e_f})[0]
    print(f"  decoder_no_cache   : {np.abs(l_f - l_q).max():.3e}"
          f"  (logits 幅值 ~{np.abs(l_f).mean():.2f})")
    agree = (l_f[0, -1].argmax() == l_q[0, -1].argmax())
    print(f"  argmax 一致        : {agree}")

    # 贪心解码一致性 + 速度（同一 fp32 encoder 输出，隔离 decoder 量化影响）
    cr_f = make_session(src / "cross_kv_init.onnx")
    cr_q = make_session(dst / "cross_kv_init.onnx")
    wc_f = make_session(src / "decoder_with_cache.onnx")
    wc_q = make_session(dst / "decoder_with_cache.onnx")
    seq_f, t_f = greedy_with_cache(cr_f, wc_f, e_f, steps)
    seq_q, t_q = greedy_with_cache(cr_q, wc_q, e_f, steps)
    n_same = sum(a == b for a, b in zip(seq_f, seq_q))
    print(f"\n== 贪心解码 {steps} 步 (with KV cache) ==")
    print(f"  token 一致率       : {n_same}/{len(seq_f)}"
          f"  ({'完全一致' if seq_f == seq_q else '存在分歧'})")
    print(f"  fp32 : {t_f:.3f}s ({t_f/steps*1000:.1f} ms/step)")
    print(f"  int8 : {t_q:.3f}s ({t_q/steps*1000:.1f} ms/step)  {t_f/t_q:.2f}x")


def layerwise_diff(src: Path, dst: Path):
    """把 decoder 每层输出挂成图输出，定位量化误差从哪层开始放大。"""
    print("\n== 逐层误差定位 (decoder_no_cache, fp32 vs int8) ==")
    rng = np.random.default_rng(0)
    mel = rng.standard_normal((1, N_MELS, 3000), dtype=np.float32)
    enc = make_session(src / "encoder.onnx").run(None, {"mel": mel})[0]
    feeds = {"tokens": np.array([PROMPT], dtype=np.int64), "encoder_hidden": enc}

    def tap_layers(path):
        m = onnx.load(str(path))
        # 取每层作用域顶层的最后一个 Add（第三个残差连接 = 层最终输出），
        # 排除子模块（self_attn/encoder_attn/fc）内部的 Add
        taps = {}
        for node in m.graph.node:
            if node.op_type != "Add":
                continue
            for out in node.output:
                for i in range(LAYERS):
                    prefix = f"/decoder/layers.{i}/"
                    if out.startswith(prefix) and "/" not in out[len(prefix):]:
                        taps.setdefault(i, []).append(out)
        chosen = {i: v[-1] for i, v in taps.items()}  # 节点序即拓扑序
        existing = {o.name for o in m.graph.output}
        for name in chosen.values():
            if name not in existing:
                m.graph.output.append(
                    onnx.helper.make_empty_tensor_value_info(name))
        return m, chosen

    m_f, taps_f = tap_layers(src / "decoder_no_cache.onnx")
    m_q, taps_q = tap_layers(dst / "decoder_no_cache.onnx")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pf, pq = Path(td) / "f.onnx", Path(td) / "q.onnx"
        onnx.save(m_f, str(pf))
        onnx.save(m_q, str(pq))
        sf, sq = make_session(pf), make_session(pq)
        out_f = dict(zip([o.name for o in sf.get_outputs()],
                         sf.run(None, feeds)))
        out_q = dict(zip([o.name for o in sq.get_outputs()],
                         sq.run(None, feeds)))

    for i in range(LAYERS):
        a, b = out_f.get(taps_f.get(i)), out_q.get(taps_q.get(i))
        if a is None or b is None or a.shape != b.shape:
            print(f"  layer {i}: 张量名不匹配，跳过 (fp32={taps_f.get(i)}, int8={taps_q.get(i)})")
            continue
        diff = np.abs(a - b)
        print(f"  layer {i} ({taps_f[i]}): max {diff.max():.3e}  mean {diff.mean():.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="../onnx")
    ap.add_argument("--out", default="../onnx_int8")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--layerwise", action="store_true")
    ap.add_argument("--skip-quant", action="store_true", help="已量化过，只做检查")
    args = ap.parse_args()

    src, dst = Path(args.onnx), Path(args.out)
    if not args.skip_quant:
        quantize_all(src, dst)
    check_outputs(src, dst, args.steps)
    if args.layerwise:
        layerwise_diff(src, dst)


if __name__ == "__main__":
    main()
