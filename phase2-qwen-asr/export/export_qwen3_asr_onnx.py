#!/usr/bin/env python
"""
Qwen3-ASR-0.6B → ONNX 导出（TorchScript 跟踪，dynamo=False，opset 17，fp32）。

拆成 5 个模型（无 cross-attention，音频嵌入直接拼 prompt，与 whisper 结构不同）：

  audio_frontend.onnx    chunks[N,1,128,100] → [N,13,896]
                         conv2d×3(stride2) + conv_out + 每块独立的正弦位置编码。
                         mel 按 100 帧(1s)分块、尾块补零到 100 在图外完成。
  audio_transformer.onnx hidden[S,896] + attn_mask[1,1,S,S] → [S,1024]
                         18 层双向 transformer + ln_post + proj1/gelu/proj2。
                         分窗注意力(8s 一窗)的 block-diagonal mask 由图外构造传入。
  embed.onnx             input_ids[1,L] → [1,L,1024]（Gather 查表，便于 int8 量化）
  decoder.onnx           inputs_embeds[1,L,1024] + past_k/v[28,1,8,P,128]
                         → hidden[1,L,1024] + present_k/v[28,1,8,P+L,128]
                         一张图同时覆盖 prefill(L>1,P=0) 与增量解码(L=1,P>0)。
  lm_head.onnx           hidden[1,1,1024] → logits[1,1,151936]
                         单独拆出：① 5.9 亿参数 fp32 全量 decoder 超 protobuf 2GB 限制；
                         ② 贪心解码只需最后一个位置的 logits，prefill 省 L×151936 矩阵乘。

导出期的两个关键事实（决定了实现方式，细节见 REPORT）：
  1. ASR 场景下 get_rope_index 给 3 路 MRoPE 的 position_ids 完全相同，
     apply_interleaved_mrope 退化为恒等变换 → decoder 可按标准 1D RoPE 重写导出。
  2. HF 的 DynamicCache/masking_utils 不可 trace → decoder 用原模块权重
     (q_proj/k_norm/...) 手写无 Cache 类的前向，position/mask 从 size() 动态推导。

用法:
  ../venv/Scripts/python export_qwen3_asr_onnx.py --model M:/models/Qwen3-ASR-0.6B --out ../onnx
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

OPSET = 17
CHUNK_FRAMES = 100          # n_window * 2，1 秒 mel
FRAMES_PER_CHUNK_OUT = 13   # 100 帧 → conv 后 13 帧
NEG = torch.finfo(torch.float32).min


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------- audio tower

class AudioFrontend(nn.Module):
    """conv 前端，按块处理。位置编码每块从 0 重新计（与原实现一致：
    pos emb 在分块 pad 之后、flatten 之前加，块内位置 0..12）。"""

    def __init__(self, at):
        super().__init__()
        self.at = at
        self.register_buffer(
            "pos", at.positional_embedding.positional_embedding[:FRAMES_PER_CHUNK_OUT].float())

    def forward(self, chunks):           # [N,1,128,100]
        at = self.at
        x = torch.nn.functional.gelu(at.conv2d1(chunks))
        x = torch.nn.functional.gelu(at.conv2d2(x))
        x = torch.nn.functional.gelu(at.conv2d3(x))   # [N,480,16,13]
        b, c, f, t = x.size()
        x = at.conv_out(x.permute(0, 3, 1, 2).reshape(b, t, c * f))  # [N,13,896]
        return x + self.pos.unsqueeze(0)


class AudioTransformer(nn.Module):
    """18 层双向 transformer。attn_mask 为加性 mask（block-diagonal 分窗），图外构造。"""

    def __init__(self, at):
        super().__init__()
        self.at = at

    def forward(self, hidden, attn_mask):    # [S,896], [1,1,S,S]
        at = self.at
        n_heads = at.layers[0].self_attn.num_heads
        head_dim = at.layers[0].self_attn.head_dim
        scaling = head_dim ** -0.5
        for layer in self.at.layers:
            residual = hidden
            x = layer.self_attn_layer_norm(hidden)
            sa = layer.self_attn
            S = x.size(0)
            q = sa.q_proj(x).reshape(S, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
            k = sa.k_proj(x).reshape(S, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
            v = sa.v_proj(x).reshape(S, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
            w = torch.matmul(q, k.transpose(2, 3)) * scaling + attn_mask
            w = torch.softmax(w, dim=-1)
            o = torch.matmul(w, v).transpose(1, 2).reshape(S, -1)
            hidden = residual + sa.out_proj(o)
            residual = hidden
            x = layer.final_layer_norm(hidden)
            x = layer.fc2(layer.activation_fn(layer.fc1(x)))
            hidden = residual + x
        hidden = at.ln_post(hidden)
        return at.proj2(at.act(at.proj1(hidden)))     # [S,1024]


# ----------------------------------------------------------------------- llm

class Embed(nn.Module):
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids):        # [1,L] → [1,L,1024]
        return self.embed_tokens(input_ids)


class Decoder(nn.Module):
    """28 层 Qwen3 decoder，手写前向：标准 1D RoPE（MRoPE 在 ASR 下退化，见模块说明）、
    GQA 8KV→16Q、q/k per-head RMSNorm、显式 KV cache 拼接、因果 mask 从 size 动态构造。"""

    def __init__(self, text_model):
        super().__init__()
        self.m = text_model
        self.register_buffer("inv_freq", text_model.rotary_emb.inv_freq.float())  # [64]

    def forward(self, inputs_embeds, past_k, past_v):
        # inputs_embeds [1,L,1024]; past_k/v [28,1,8,P,128]
        L = inputs_embeds.size(1)
        P = past_k.size(3)
        n_layers = past_k.size(0)

        pos = torch.arange(P, P + L, dtype=torch.float32)          # [L]
        freqs = pos[:, None] * self.inv_freq[None, :]              # [L,64]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]    # [1,1,L,128]

        # 因果 mask：query 绝对位置 P+i 只看 key 位置 <= P+i
        q_pos = torch.arange(L)[:, None] + P
        k_pos = torch.arange(P + L)[None, :]
        mask = torch.where(k_pos > q_pos,
                           torch.full((), NEG), torch.zeros(()))[None, None]  # [1,1,L,P+L]

        hidden = inputs_embeds
        new_ks, new_vs = [], []
        for i, layer in enumerate(self.m.layers):
            residual = hidden
            x = layer.input_layernorm(hidden)
            sa = layer.self_attn
            q = sa.q_norm(sa.q_proj(x).view(1, L, -1, sa.head_dim)).transpose(1, 2)  # [1,16,L,128]
            k = sa.k_norm(sa.k_proj(x).view(1, L, -1, sa.head_dim)).transpose(1, 2)  # [1,8,L,128]
            v = sa.v_proj(x).view(1, L, -1, sa.head_dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            k = torch.cat([past_k[i], k], dim=2)        # [1,8,P+L,128]
            v = torch.cat([past_v[i], v], dim=2)
            new_ks.append(k)
            new_vs.append(v)
            # GQA: 8 KV 头扩到 16
            kr = k.unsqueeze(2).expand(1, 8, 2, k.size(2), sa.head_dim).reshape(1, 16, -1, sa.head_dim)
            vr = v.unsqueeze(2).expand(1, 8, 2, v.size(2), sa.head_dim).reshape(1, 16, -1, sa.head_dim)
            w = torch.matmul(q, kr.transpose(2, 3)) * sa.scaling + mask
            w = torch.softmax(w, dim=-1)
            o = torch.matmul(w, vr).transpose(1, 2).reshape(1, L, -1)
            hidden = residual + sa.o_proj(o)
            residual = hidden
            x = layer.post_attention_layernorm(hidden)
            x = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x))
            hidden = residual + x

        hidden = self.m.norm(hidden)                    # [1,L,1024]
        return hidden, torch.stack(new_ks), torch.stack(new_vs)


class LMHead(nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden):           # [1,1,1024] → [1,1,151936]
        return self.lm_head(hidden)


# -------------------------------------------------------------------- export

def export(model_dir: str, out_dir: Path):
    from qwen_asr import Qwen3ASRModel
    print("加载模型 (fp32, cpu, eager attn) ...", flush=True)
    wrapper = Qwen3ASRModel.from_pretrained(
        model_dir, dtype=torch.float32, device_map="cpu",
        attn_implementation="eager")
    thinker = wrapper.model.thinker
    thinker.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(opset_version=OPSET, dynamo=False, do_constant_folding=True)

    # 1. audio_frontend
    fe = AudioFrontend(thinker.audio_tower).eval()
    chunks = torch.randn(3, 1, 128, CHUNK_FRAMES)
    torch.onnx.export(fe, (chunks,), out_dir / "audio_frontend.onnx",
                      input_names=["chunks"], output_names=["chunk_embeds"],
                      dynamic_axes={"chunks": {0: "n"}, "chunk_embeds": {0: "n"}}, **kw)
    print("  audio_frontend.onnx done")

    # 2. audio_transformer
    tr = AudioTransformer(thinker.audio_tower).eval()
    S = 39
    hidden = torch.randn(S, 896)
    amask = torch.zeros(1, 1, S, S)
    torch.onnx.export(tr, (hidden, amask), out_dir / "audio_transformer.onnx",
                      input_names=["hidden", "attn_mask"], output_names=["audio_embeds"],
                      dynamic_axes={"hidden": {0: "s"},
                                    "attn_mask": {2: "s", 3: "s"},
                                    "audio_embeds": {0: "s"}}, **kw)
    print("  audio_transformer.onnx done")

    # 3. embed
    emb = Embed(thinker.model.embed_tokens).eval()
    ids = torch.tensor([[151644, 8948, 198]], dtype=torch.int64)
    torch.onnx.export(emb, (ids,), out_dir / "embed.onnx",
                      input_names=["input_ids"], output_names=["embeds"],
                      dynamic_axes={"input_ids": {1: "l"}, "embeds": {1: "l"}}, **kw)
    print("  embed.onnx done")

    # 4. decoder（prefill 形状导出，verify 用 L=1/P>0 检验动态性）
    dec = Decoder(thinker.model).eval()
    L, P = 8, 5
    x = torch.randn(1, L, 1024)
    pk = torch.randn(28, 1, 8, P, 128)
    pv = torch.randn(28, 1, 8, P, 128)
    torch.onnx.export(dec, (x, pk, pv), out_dir / "decoder.onnx",
                      input_names=["inputs_embeds", "past_k", "past_v"],
                      output_names=["hidden", "present_k", "present_v"],
                      dynamic_axes={"inputs_embeds": {1: "l"},
                                    "past_k": {3: "p"}, "past_v": {3: "p"},
                                    "hidden": {1: "l"},
                                    "present_k": {3: "pl"}, "present_v": {3: "pl"}}, **kw)
    print("  decoder.onnx done")

    # 5. lm_head
    lh = LMHead(thinker.lm_head).eval()
    h = torch.randn(1, 1, 1024)
    torch.onnx.export(lh, (h,), out_dir / "lm_head.onnx",
                      input_names=["hidden"], output_names=["logits"], **kw)
    print("  lm_head.onnx done")

    for f in sorted(out_dir.glob("*.onnx")):
        print(f"  {f.name:24s} {f.stat().st_size/2**20:8.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="M:/models/Qwen3-ASR-0.6B")
    ap.add_argument("--out", default="../onnx")
    args = ap.parse_args()
    with torch.no_grad():
        export(args.model, Path(args.out))


if __name__ == "__main__":
    main()
