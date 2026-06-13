#!/usr/bin/env python
"""
Qwen3-ASR ONNX 动态 INT8 量化 + 量化前后精度对齐。

  1. 5 个模型 quantize_dynamic（MatMul/Gemm/Gather 权重 int8，per-channel）
     - audio_frontend 的 3 个 Conv2d 保持 fp32（ORT CPU EP 无 ConvInteger）
     - embed/lm_head 的 Gather/MatMul 量化 = 权重表 int8，体积约减半以上
  2. 真实音频下 fp32 vs int8 各级输出 max abs diff
  3. 端到端贪心 token 一致性 + 文本对比 + 速度
  4. --layerwise: decoder 28 层逐层输出误差定位（误差从哪层开始放大）

用法:
  ../venv/Scripts/python quantize_int8.py --onnx ../onnx --out ../onnx_int8 [--layerwise] [--skip-quant]
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import soundfile as sf
import pyarrow.parquet as pq_
from onnxruntime.quantization import QuantType, quantize_dynamic

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "../export"))
from onnx_pipeline import QwenASROnnxPipeline, make_session, N_LAYERS  # noqa: E402

MODELS = ["audio_frontend", "audio_transformer", "embed", "decoder", "lm_head"]
PARQUET = HERE / "../../phase1-asr/data/aishell1_test-00000.parquet"


def load_audio(num: int, seed: int = 0):
    table = pq_.ParquetFile(PARQUET).read(columns=["context", "answer"])
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(table.num_rows, size=num, replace=False))
    rows = table.take(idx).to_pylist()
    return [(sf.read(io.BytesIO(r["context"]["bytes"]), dtype="float32")[0], r["answer"])
            for r in rows]


def make_mel(audio):
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained("M:/models/Qwen3-ASR-0.6B")
    feats = fe(audio, sampling_rate=16000, return_attention_mask=True,
               padding="do_not_pad", return_tensors="np")
    return feats.input_features[0].astype(np.float32)


def quantize_all(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    print("== 动态 INT8 量化 (MatMul/Gemm/Gather, per-channel, QInt8) ==", flush=True)
    for name in MODELS:
        fp32, int8 = src / f"{name}.onnx", dst / f"{name}.onnx"
        t0 = time.perf_counter()
        quantize_dynamic(
            str(fp32), str(int8), weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm", "Gather"],
            per_channel=True,
        )
        s0, s1 = fp32.stat().st_size / 2**20, int8.stat().st_size / 2**20
        print(f"  {name:18s} {s0:8.1f} MB -> {s1:8.1f} MB ({s1/s0*100:.0f}%)"
              f"  [{time.perf_counter()-t0:.0f}s]", flush=True)


def check_outputs(src: Path, dst: Path, num: int):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("M:/models/Qwen3-ASR-0.6B")
    pipe_f = QwenASROnnxPipeline(src)
    pipe_q = QwenASROnnxPipeline(dst)
    samples = load_audio(num)

    print("\n== fp32 vs int8 对齐（真实音频）==", flush=True)
    all_same = True
    for i, (audio, answer) in enumerate(samples):
        mel = make_mel(audio)
        dur = len(audio) / 16000

        # 音频编码器
        e_f = pipe_f.encode_audio(mel)
        e_q = pipe_q.encode_audio(mel)
        d = np.abs(e_f - e_q)
        cos = (e_f * e_q).sum() / (np.linalg.norm(e_f) * np.linalg.norm(e_q))
        print(f"\n样本 {i} ({dur:.1f}s)  ref={answer}")
        print(f"  audio_encoder diff: max {d.max():.3e}  mean {d.mean():.3e}  cos {cos:.6f}")

        # 端到端贪心
        toks_f, tf1, tft = pipe_f.transcribe(mel)
        toks_q, tq1, tqt = pipe_q.transcribe(mel)
        same = toks_f == toks_q
        all_same &= same
        n_pre = sum(a == b for a, b in zip(toks_f, toks_q))
        print(f"  greedy: fp32 {len(toks_f)} toks vs int8 {len(toks_q)} toks  "
              f"前缀一致 {n_pre}  {'完全一致' if same else '存在分歧'}")
        print(f"  fp32 文本: {tok.decode(toks_f)}")
        if not same:
            print(f"  int8 文本: {tok.decode(toks_q)}")
        print(f"  耗时: fp32 {tft:.2f}s (RTF {tft/dur:.2f})  "
              f"int8 {tqt:.2f}s (RTF {tqt/dur:.2f})  加速 {tft/tqt:.2f}x")

    print(f"\n{'✓ 全部样本 int8 贪心序列与 fp32 一致' if all_same else '△ int8 与 fp32 存在 token 分歧（CER 评测见 bench）'}")


def layerwise_diff(src: Path, dst: Path):
    """decoder 28 层逐层输出误差定位。
    TorchScript trace 的作用域命名：layer i 的 input_layernorm → /input_layernorm_{i}/
    （layer 0 无后缀），最终 RMSNorm → /norm/。
    layer i 的输出 = layer i+1 input_layernorm 的输入（最后一层 = /norm/ 的输入），
    该张量名在 fp32/int8 图中一致（量化只改 MatMul 权重，不改激活张量名）。"""
    print("\n== decoder 逐层误差定位 (fp32 vs int8) ==", flush=True)

    def find_taps(path):
        m = onnx.load(str(path), load_external_data=False)
        taps = {}
        for node in m.graph.node:
            if not node.input:
                continue
            if node.name.startswith("/input_layernorm"):
                scope = node.name.split("/")[1]          # input_layernorm[_i]
                suffix = scope[len("input_layernorm"):]
                i = int(suffix[1:]) if suffix else 0
                if i >= 1:
                    taps.setdefault(i - 1, node.input[0])
            elif node.name.startswith("/norm/"):
                taps.setdefault(N_LAYERS - 1, node.input[0])
        return taps

    taps = find_taps(src / "decoder.onnx")
    if len(taps) < N_LAYERS:
        print(f"  仅定位到 {len(taps)}/{N_LAYERS} 层 tap 点: {sorted(taps)}")

    def tap_model(path, out_path):
        m = onnx.load(str(path))
        existing = {o.name for o in m.graph.output}
        for name in taps.values():
            if name not in existing:
                m.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
        onnx.save(m, str(out_path))

    import tempfile
    audio, _ = load_audio(1)[0]
    mel = make_mel(audio)
    pipe = QwenASROnnxPipeline(src)
    prompt = pipe.build_prompt_embeds(pipe.encode_audio(mel))
    pk = np.zeros((N_LAYERS, 1, 8, 0, 128), dtype=np.float32)
    feeds = {"inputs_embeds": prompt, "past_k": pk, "past_v": pk.copy()}

    with tempfile.TemporaryDirectory() as td:
        pf, pq2 = Path(td) / "f.onnx", Path(td) / "q.onnx"
        tap_model(src / "decoder.onnx", pf)
        tap_model(dst / "decoder.onnx", pq2)
        for tag, p in (("fp32", pf), ("int8", pq2)):
            s = make_session(p)
            outs = dict(zip([o.name for o in s.get_outputs()], s.run(None, feeds)))
            if tag == "fp32":
                ref = outs
            else:
                for i in sorted(taps):
                    a, b = ref[taps[i]], outs[taps[i]]
                    d = np.abs(a - b)
                    print(f"  layer {i:2d}: max {d.max():.3e}  mean {d.mean():.3e}"
                          f"  (幅值 ~{np.abs(a).mean():.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="../onnx")
    ap.add_argument("--out", default="../onnx_int8")
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--layerwise", action="store_true")
    ap.add_argument("--skip-quant", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.onnx), Path(args.out)
    if not args.skip_quant:
        quantize_all(src, dst)
    check_outputs(src, dst, args.num)
    if args.layerwise:
        layerwise_diff(src, dst)


if __name__ == "__main__":
    main()
