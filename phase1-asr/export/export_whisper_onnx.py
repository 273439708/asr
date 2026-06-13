#!/usr/bin/env python
"""
Whisper PyTorch → ONNX 导出脚本。

导出 3 个模型：
  1. encoder.onnx               — mel(80x3000) → hidden states
  2. decoder_no_cache.onnx      — 每步重算全部历史 token（无 KV cache，作对比基线）
  3. decoder_with_cache.onnx    — 带 self-attn KV cache 增量解码 + cross-attn KV 预计算

用法:
  python export_whisper_onnx.py --model openai/whisper-base --out ../onnx
"""

import argparse
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration


def patch_mask_for_onnx_export():
    """
    transformers>=4.53 的 masking_utils 用 torch.vmap 构建因果 mask，
    TorchScript 跟踪导出（dynamo=False）不支持 vmap（报 invalid unordered_map key）。
    这里用纯广播实现替换 sdpa/eager 两个 mask 接口（导出场景无 padding，语义等价）。
    """
    from transformers import masking_utils

    def sdpa_mask_no_vmap(batch_size, cache_position, kv_length, kv_offset=0,
                          mask_function=None, attention_mask=None,
                          dtype=torch.float32, **kwargs):
        kv_arange = torch.arange(kv_length, device=cache_position.device) + kv_offset
        causal = kv_arange[None, :] <= cache_position[:, None]        # [q_len, kv_len]
        mask = causal[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            mask = mask & attention_mask[:, None, None, -kv_length:].bool()
        return mask

    def eager_mask_no_vmap(batch_size, cache_position, kv_length, kv_offset=0,
                           mask_function=None, attention_mask=None,
                           dtype=torch.float32, **kwargs):
        bool_mask = sdpa_mask_no_vmap(
            batch_size, cache_position, kv_length, kv_offset,
            mask_function, attention_mask, dtype)
        zero = torch.tensor(0.0, dtype=dtype, device=cache_position.device)
        return torch.where(bool_mask, zero, torch.finfo(dtype).min)

    reg = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    reg["sdpa"] = sdpa_mask_no_vmap
    reg["eager"] = eager_mask_no_vmap


class EncoderWrapper(torch.nn.Module):
    """mel → encoder hidden states"""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel).last_hidden_state


class DecoderNoCache(torch.nn.Module):
    """无 KV cache：输入全部已生成 token，每步重算。O(n^2) 复杂度的朴素解码。"""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    def forward(self, tokens: torch.Tensor, encoder_hidden: torch.Tensor) -> torch.Tensor:
        out = self.decoder(
            input_ids=tokens,
            encoder_hidden_states=encoder_hidden,
            use_cache=False,
        ).last_hidden_state
        return self.proj_out(out)  # [B, T, vocab]


class DecoderWithCache(torch.nn.Module):
    """
    带 KV cache 增量解码：每步只输入 1 个新 token + 历史 self-attn K/V。
    cross-attn K/V 与 token 无关，由 encoder 输出预计算一次后重复使用。

    输入:
      token          [B, 1]
      self_k/self_v  [layers, B, heads, past_len, head_dim]
      cross_k/cross_v[layers, B, heads, src_len, head_dim]
    输出:
      logits         [B, 1, vocab]
      new_self_k/v   past_len + 1
    """

    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out
        cfg = model.config
        self.num_layers = cfg.decoder_layers
        self.num_heads = cfg.decoder_attention_heads
        self.head_dim = cfg.d_model // cfg.decoder_attention_heads

    def forward(self, token, self_k, self_v, cross_k, cross_v):
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        past_len = self_k.shape[3]
        self_cache = DynamicCache()
        cross_cache = DynamicCache()
        for i in range(self.num_layers):
            self_cache.update(self_k[i], self_v[i], i)
            cross_cache.update(cross_k[i], cross_v[i], i)
        cache = EncoderDecoderCache(self_cache, cross_cache)

        # 注意：decoder 层用 `if encoder_hidden_states is not None` 决定是否执行
        # cross-attn 块（modeling_whisper.py:516）。不传则 cross-attn 被整体跳过、
        # cross_k/v 输入被剪枝，导出的模型不依赖音频 → 必须传一个非 None 的张量。
        # is_updated=True 时 K/V 直接取 cache，该张量内容不会参与计算，
        # 这里从 cross_k 还原出正确形状 [B, src_len, d_model] 作占位。
        bsz = token.shape[0]
        src_len = cross_k.shape[3]
        dummy_enc = cross_k[0].permute(0, 2, 1, 3).reshape(
            bsz, src_len, self.num_heads * self.head_dim)

        out = self.decoder(
            input_ids=token,
            encoder_hidden_states=dummy_enc,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(past_len, past_len + 1, device=token.device),
        )
        logits = self.proj_out(out.last_hidden_state)

        new_cache = out.past_key_values.self_attention_cache
        new_k = torch.stack([new_cache.layers[i].keys for i in range(self.num_layers)])
        new_v = torch.stack([new_cache.layers[i].values for i in range(self.num_layers)])
        return logits, new_k, new_v


class CrossKVInit(torch.nn.Module):
    """encoder 输出 → 各层 cross-attn 的 K/V（一次性预计算）。"""

    def __init__(self, model):
        super().__init__()
        self.layers = model.model.decoder.layers
        cfg = model.config
        self.num_heads = cfg.decoder_attention_heads
        self.head_dim = cfg.d_model // cfg.decoder_attention_heads

    def forward(self, encoder_hidden: torch.Tensor):
        ks, vs = [], []
        bsz, src_len, _ = encoder_hidden.shape
        for layer in self.layers:
            attn = layer.encoder_attn
            k = attn.k_proj(encoder_hidden)
            v = attn.v_proj(encoder_hidden)
            k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
            ks.append(k)
            vs.append(v)
        return torch.stack(ks), torch.stack(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--out", default="../onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_mask_for_onnx_export()

    print(f"loading {args.model} ...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.eval()
    cfg = model.config
    n_mels = cfg.num_mel_bins
    layers = cfg.decoder_layers
    heads = cfg.decoder_attention_heads
    head_dim = cfg.d_model // heads
    print(f"  mels={n_mels} d_model={cfg.d_model} dec_layers={layers} heads={heads}")

    # ---------- 1. encoder ----------
    mel = torch.randn(1, n_mels, 3000)
    enc = EncoderWrapper(model)
    with torch.no_grad():
        enc_out = enc(mel)
    torch.onnx.export(
        enc, (mel,), out_dir / "encoder.onnx",
        input_names=["mel"], output_names=["encoder_hidden"],
        dynamic_axes={"mel": {0: "batch"}, "encoder_hidden": {0: "batch"}},
        opset_version=args.opset, dynamo=False,
    )
    print(f"[1/4] encoder.onnx  hidden={tuple(enc_out.shape)}")

    # ---------- 2. decoder（无 cache） ----------
    tokens = torch.tensor([[50258, 50260, 50359, 50363]], dtype=torch.long)
    dec_nc = DecoderNoCache(model)
    torch.onnx.export(
        dec_nc, (tokens, enc_out), out_dir / "decoder_no_cache.onnx",
        input_names=["tokens", "encoder_hidden"], output_names=["logits"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "seq"},
            "encoder_hidden": {0: "batch"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=args.opset, dynamo=False,
    )
    print("[2/4] decoder_no_cache.onnx")

    # ---------- 3. cross-attn KV 预计算 ----------
    cross_init = CrossKVInit(model)
    with torch.no_grad():
        cross_k, cross_v = cross_init(enc_out)
    torch.onnx.export(
        cross_init, (enc_out,), out_dir / "cross_kv_init.onnx",
        input_names=["encoder_hidden"], output_names=["cross_k", "cross_v"],
        dynamic_axes={"encoder_hidden": {0: "batch"},
                      "cross_k": {1: "batch"}, "cross_v": {1: "batch"}},
        opset_version=args.opset, dynamo=False,
    )
    print(f"[3/4] cross_kv_init.onnx  cross_k={tuple(cross_k.shape)}")

    # ---------- 4. decoder（带 KV cache，单步） ----------
    token = torch.tensor([[50258]], dtype=torch.long)
    past = 4
    self_k = torch.randn(layers, 1, heads, past, head_dim)
    self_v = torch.randn(layers, 1, heads, past, head_dim)
    dec_c = DecoderWithCache(model)
    torch.onnx.export(
        dec_c, (token, self_k, self_v, cross_k, cross_v),
        out_dir / "decoder_with_cache.onnx",
        input_names=["token", "self_k", "self_v", "cross_k", "cross_v"],
        output_names=["logits", "new_self_k", "new_self_v"],
        dynamic_axes={
            "token": {0: "batch"},
            "self_k": {1: "batch", 3: "past_len"},
            "self_v": {1: "batch", 3: "past_len"},
            "cross_k": {1: "batch"}, "cross_v": {1: "batch"},
            "logits": {0: "batch"},
            "new_self_k": {1: "batch", 3: "new_past_len"},
            "new_self_v": {1: "batch", 3: "new_past_len"},
        },
        opset_version=args.opset, dynamo=False,
    )
    print("[4/4] decoder_with_cache.onnx")

    for f in sorted(out_dir.glob("*.onnx")):
        print(f"  {f.name:28s} {f.stat().st_size/1e6:8.1f} MB")
    print("done.")


if __name__ == "__main__":
    main()
