#!/usr/bin/env python
"""
ONNX vs PyTorch 对齐验证（fp32, CPU）：
  1. 音频编码器输出 max abs diff（两条不同长度样本 → 同时检验动态轴/尾块处理）
  2. 端到端贪心 token 序列对比（ONNX 五模型管线 vs HF thinker.generate）

用法:
  ../venv/Scripts/python verify_onnx.py --onnx ../onnx [--num 3]
"""

import argparse
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
import torch

from onnx_pipeline import QwenASROnnxPipeline

HERE = Path(__file__).parent
PARQUET = HERE / "../../phase1-asr/data/aishell1_test-00000.parquet"
MODEL = "M:/models/Qwen3-ASR-0.6B"


def load_samples(num: int, seed: int = 0):
    pf = pq.ParquetFile(PARQUET)
    table = pf.read(columns=["context", "answer"])
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(table.num_rows, size=num, replace=False))
    rows = table.take(idx).to_pylist()
    out = []
    for r in rows:
        audio, sr = sf.read(io.BytesIO(r["context"]["bytes"]), dtype="float32")
        out.append((audio, r["answer"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="../onnx")
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    from qwen_asr import Qwen3ASRModel
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    print("加载 PyTorch 模型 (fp32, cpu, eager) ...", flush=True)
    wrapper = Qwen3ASRModel.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    thinker = wrapper.model.thinker.eval()
    processor = Qwen3ASRProcessor.from_pretrained(MODEL)
    fe, tok = processor.feature_extractor, processor.tokenizer

    pipe = QwenASROnnxPipeline(args.onnx)
    samples = load_samples(args.num)

    msgs = [{"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": "x"}]}]
    prompt_txt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    all_match = True
    for i, (audio, answer) in enumerate(samples):
        dur = len(audio) / 16000
        feats = fe(audio, sampling_rate=16000, return_attention_mask=True,
                   padding="do_not_pad", return_tensors="np")
        mel = feats.input_features[0].astype(np.float32)        # [128,T]
        T = mel.shape[1]
        print(f"\n== 样本 {i}: {dur:.1f}s  mel T={T}  ref={answer}")

        # 1) 音频编码器对齐
        with torch.no_grad():
            ref_emb = thinker.audio_tower(
                torch.from_numpy(mel), feature_lens=torch.tensor([T])
            ).last_hidden_state.numpy()
        onnx_emb = pipe.encode_audio(mel)
        diff = np.abs(ref_emb - onnx_emb).max()
        print(f"  audio_encoder: shape {onnx_emb.shape}  max_diff={diff:.2e}")

        # 2) 端到端贪心 token 对比
        inputs = processor(text=[prompt_txt], audio=[audio], return_tensors="pt", padding=True)
        with torch.no_grad():
            gen = thinker.generate(**inputs, do_sample=False, max_new_tokens=128,
                                   eos_token_id=151645)  # <|im_end|>
        ref_toks = gen[0, inputs.input_ids.shape[1]:].tolist()
        if 151645 in ref_toks:  # 截断到首个 <|im_end|>（generate 可能继续生成）
            ref_toks = ref_toks[:ref_toks.index(151645)]

        onnx_toks, t_first, t_total = pipe.transcribe(mel)
        match = onnx_toks == ref_toks
        all_match &= match
        print(f"  greedy tokens: pytorch {len(ref_toks)} vs onnx {len(onnx_toks)}  "
              f"{'完全一致' if match else '不一致!'}")
        print(f"  onnx 文本: {tok.decode(onnx_toks, skip_special_tokens=False)}")
        if not match:
            print(f"  pt   文本: {tok.decode(ref_toks, skip_special_tokens=False)}")
            print(f"  pt   toks: {ref_toks[:30]}")
            print(f"  onnx toks: {onnx_toks[:30]}")
        print(f"  onnx 耗时: 首token {t_first*1000:.0f}ms  总 {t_total:.2f}s  RTF {t_total/dur:.3f}")

    print(f"\n{'✓ 全部样本贪心序列与 PyTorch 完全一致' if all_match else '✗ 存在不一致，需排查'}")


if __name__ == "__main__":
    main()
