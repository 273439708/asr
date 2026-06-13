#!/usr/bin/env python
"""
Qwen3-ASR ONNX 五模型推理管线（fp32/int8 通用），供 verify 与 bench 复用。

管线: mel[128,T] → 分块(100帧/块,尾块补零) → audio_frontend → 去尾块无效帧
     → block-diagonal 分窗 mask → audio_transformer → audio_embeds[S,1024]
     → 拼 prompt（embed 查表 + 占位替换）→ decoder prefill → 贪心增量解码

图外逻辑严格对齐 modeling_qwen3_asr.py：
  - 块内卷积输出有效帧数 = _get_feat_extract_output_lengths（逐级 ceil/2 三次）
  - 注意力分窗 window_aftercnn = 13 * (n_window_infer=800 / 100) = 104 帧（8s/窗）
"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

CHUNK = 100          # mel 帧/块
CHUNK_OUT = 13       # conv 后帧/块
WINDOW = 104         # 注意力分窗（conv 后帧数, 8s）
NEG = np.finfo(np.float32).min

# prompt: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
#         [audio]<|audio_end|><|im_end|>\n<|im_start|>assistant\n
PROMPT_HEAD = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
PROMPT_TAIL = [151670, 151645, 198, 151644, 77091, 198]
EOS = 151645         # <|im_end|>
N_LAYERS, N_KV, HEAD_DIM = 28, 8, 128


def conv_out_len(t: int) -> int:
    """单块 t(<=100) 帧 mel 经 3 层 stride-2 conv 的有效输出帧数。"""
    for _ in range(3):
        t = (t - 1) // 2 + 1
    return t


def make_session(path, providers=None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(
        str(path), so, providers=providers or ["CPUExecutionProvider"])


class QwenASROnnxPipeline:
    def __init__(self, model_dir, providers=None):
        d = Path(model_dir)
        self.frontend = make_session(d / "audio_frontend.onnx", providers)
        self.transformer = make_session(d / "audio_transformer.onnx", providers)
        self.embed = make_session(d / "embed.onnx", providers)
        self.decoder = make_session(d / "decoder.onnx", providers)
        self.lm_head = make_session(d / "lm_head.onnx", providers)

    # ----------------------------------------------------------- audio tower
    def encode_audio(self, mel: np.ndarray) -> np.ndarray:
        """mel [128, T] → audio_embeds [S, 1024]"""
        T = mel.shape[1]
        n_full, tail = T // CHUNK, T % CHUNK
        n = n_full + (1 if tail else 0)
        padded = np.zeros((n * CHUNK, 128), dtype=np.float32)
        padded[:T] = mel.T
        chunks = padded.reshape(n, CHUNK, 128).transpose(0, 2, 1)[:, None]  # [N,1,128,100]

        chunk_embeds = self.frontend.run(None, {"chunks": chunks})[0]       # [N,13,896]

        valid = [CHUNK_OUT] * n_full + ([conv_out_len(tail)] if tail else [])
        hidden = np.concatenate([chunk_embeds[i, :v] for i, v in enumerate(valid)])  # [S,896]

        S = hidden.shape[0]
        mask = np.full((1, 1, S, S), NEG, dtype=np.float32)
        start = 0
        while start < S:
            end = min(start + WINDOW, S)
            mask[..., start:end, start:end] = 0.0
            start = end
        return self.transformer.run(None, {"hidden": hidden, "attn_mask": mask})[0]

    # ------------------------------------------------------------------ llm
    def embed_ids(self, ids) -> np.ndarray:
        return self.embed.run(
            None, {"input_ids": np.asarray([ids], dtype=np.int64)})[0]      # [1,L,1024]

    def build_prompt_embeds(self, audio_embeds: np.ndarray) -> np.ndarray:
        head = self.embed_ids(PROMPT_HEAD)
        tail = self.embed_ids(PROMPT_TAIL)
        return np.concatenate([head, audio_embeds[None], tail], axis=1)

    def greedy(self, prompt_embeds: np.ndarray, max_tokens: int = 256):
        """返回 (生成 token list 不含 EOS, 首 token 延迟, 总耗时)。"""
        t0 = time.perf_counter()
        pk = np.zeros((N_LAYERS, 1, N_KV, 0, HEAD_DIM), dtype=np.float32)
        pv = np.zeros_like(pk)
        hidden, pk, pv = self.decoder.run(None, {
            "inputs_embeds": prompt_embeds, "past_k": pk, "past_v": pv})
        logits = self.lm_head.run(None, {"hidden": hidden[:, -1:]})[0]
        nxt = int(logits[0, -1].argmax())
        t_first = time.perf_counter() - t0
        out = []
        while nxt != EOS and len(out) < max_tokens:
            out.append(nxt)
            x = self.embed_ids([nxt])
            hidden, pk, pv = self.decoder.run(None, {
                "inputs_embeds": x, "past_k": pk, "past_v": pv})
            logits = self.lm_head.run(None, {"hidden": hidden[:, -1:]})[0]
            nxt = int(logits[0, -1].argmax())
        return out, t_first, time.perf_counter() - t0

    def transcribe(self, mel: np.ndarray, max_tokens: int = 256):
        """mel [128,T] → (token list, 首token延迟, 总耗时)。耗时含音频编码。"""
        t0 = time.perf_counter()
        audio_embeds = self.encode_audio(mel)
        prompt = self.build_prompt_embeds(audio_embeds)
        t_pre = time.perf_counter() - t0
        toks, t_first, _ = self.greedy(prompt, max_tokens)
        return toks, t_pre + t_first, time.perf_counter() - t0
