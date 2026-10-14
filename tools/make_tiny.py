#!/usr/bin/env python3
"""Write a small random checkpoint in the original fishaudio/drama-3 layout (FP8 experts).

Real widths, fewer backbone layers and experts, so every engine code path runs
without the 42 GB download. Used only for engine-vs-reference tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil

import torch
from safetensors.torch import save_file


def lin(n: int, k: int, gain: float = 1.0) -> torch.Tensor:
    return (torch.randn(n, k) * gain / math.sqrt(k)).to(torch.bfloat16)


def fp8_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    b = 128
    nb, kb = (n + b - 1) // b, (k + b - 1) // b
    pad = torch.zeros(nb * b, kb * b)
    pad[:n, :k] = w.float()
    blocks = pad.reshape(nb, b, kb, b)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-8)
    scale = amax / 448.0
    q = (blocks / scale[:, None, :, None]).reshape(nb * b, kb * b)[:n, :k]
    return q.to(torch.float8_e4m3fn), scale.float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("real_config", help="config.json of the real model")
    ap.add_argument("dst")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    with open(args.real_config) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    tc["num_hidden_layers"] = args.layers
    tc["layer_types"] = tc["layer_types"][: args.layers]
    tc["mlp_layer_types"] = tc["mlp_layer_types"][: args.layers]
    tc["num_experts"] = args.experts
    dc = cfg["audio_decoder_config"]
    H = tc["hidden_size"]

    t: dict[str, torch.Tensor] = {}

    def norm(n: int) -> torch.Tensor:
        return (torch.randn(n) * 0.1).to(torch.bfloat16)

    p = "model.language_model."
    t[p + "embed_tokens.weight"] = (torch.randn(tc["vocab_size"], H) * 0.5).to(torch.bfloat16)
    t["lm_head.weight"] = lin(tc["vocab_size"], H, 2.0)
    t[p + "norm.weight"] = norm(H)
    nk, nv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    dk, dv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    for i in range(args.layers):
        lp = f"{p}layers.{i}."
        t[lp + "input_layernorm.weight"] = norm(H)
        t[lp + "post_attention_layernorm.weight"] = norm(H)
        if tc["layer_types"][i] == "linear_attention":
            la = lp + "linear_attn."
            t[la + "in_proj_qkv.weight"] = lin(2 * nk * dk + nv * dv, H)
            t[la + "in_proj_z.weight"] = lin(nv * dv, H)
            t[la + "in_proj_b.weight"] = lin(nv, H)
            t[la + "in_proj_a.weight"] = lin(nv, H)
            t[la + "out_proj.weight"] = lin(H, nv * dv, 0.5)
            t[la + "conv1d.weight"] = (torch.randn(2 * nk * dk + nv * dv, 1, 4) * 0.5).to(torch.bfloat16)
            t[la + "A_log"] = torch.log(torch.empty(nv).uniform_(1, 16)).float()
            t[la + "dt_bias"] = (torch.randn(nv) * 0.5).float()
            t[la + "norm.weight"] = (1.0 + torch.randn(dv) * 0.1).float()
        else:
            sa = lp + "self_attn."
            h, hkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
            t[sa + "q_proj.weight"] = lin(2 * h * hd, H)
            t[sa + "k_proj.weight"] = lin(hkv * hd, H)
            t[sa + "v_proj.weight"] = lin(hkv * hd, H)
            t[sa + "o_proj.weight"] = lin(H, h * hd, 0.5)
            t[sa + "q_norm.weight"] = norm(hd)
            t[sa + "k_norm.weight"] = norm(hd)
        mp = lp + "mlp."
        t[mp + "gate.weight"] = lin(args.experts, H, 4.0)
        t[mp + "shared_expert_gate.weight"] = lin(1, H)
        si = tc["shared_expert_intermediate_size"]
        t[mp + "shared_expert.gate_proj.weight"] = lin(si, H)
        t[mp + "shared_expert.up_proj.weight"] = lin(si, H)
        t[mp + "shared_expert.down_proj.weight"] = lin(H, si, 0.5)
        mi = tc["moe_intermediate_size"]
        for e in range(args.experts):
            for name, shape in (("gate_proj", (mi, H)), ("up_proj", (mi, H)), ("down_proj", (H, mi))):
                q, s = fp8_block(lin(*shape, 1.0 if name != "down_proj" else 0.5).float())
                t[f"{mp}experts.{e}.{name}.weight"] = q
                t[f"{mp}experts.{e}.{name}.weight_scale_inv"] = s

    D = dc["dim"]
    cb, ncb = dc["vocab_size"], dc["num_codebooks"]
    t["audio_decoder.codebook_embeddings.weight"] = (torch.randn(cb * ncb, D) * 0.3).to(torch.bfloat16)
    Hh, Hkv, hd = dc["n_head"], dc["n_local_heads"], dc["head_dim"]
    names = [f"audio_decoder_mtp_shared.layers.{j}." for j in range(dc["shared_enrichment_layers"])]
    names += [f"audio_decoder_mtp.layers.{j}." for j in range(ncb - 1)]
    for lp in names:
        emb = "semantic_embedding" if "shared" in lp else "codebook_embedding"
        t[lp + emb + ".weight"] = (torch.randn(cb, D) * 0.5).to(torch.bfloat16)
        for n in ("enorm", "hnorm", "final_layernorm"):
            t[lp + n + ".weight"] = norm(D)
        t[lp + "eh_proj.weight"] = lin(D, 2 * D)
        tl = lp + "transformer_layer."
        t[tl + "attention_norm.weight"] = norm(D)
        t[tl + "ffn_norm.weight"] = norm(D)
        t[tl + "attention.q_norm.weight"] = norm(hd)
        t[tl + "attention.k_norm.weight"] = norm(hd)
        t[tl + "attention.wqkv.weight"] = lin(2 * Hh * hd + 2 * Hkv * hd, D)
        t[tl + "attention.wo.weight"] = lin(D, Hh * hd, 0.5)
        inter = dc["intermediate_size"]
        t[tl + "feed_forward.w1.weight"] = lin(inter, D)
        t[tl + "feed_forward.w3.weight"] = lin(inter, D)
        t[tl + "feed_forward.w2.weight"] = lin(D, inter, 0.5)
    for j in range(ncb - 1):
        t[f"audio_decoder_mtp.heads.{j}.weight"] = lin(cb, D, 3.0)

    os.makedirs(args.dst, exist_ok=True)
    keys = sorted(t)
    half = len(keys) // 2
    shards = {"model-00001-of-00002.safetensors": keys[:half], "model-00002-of-00002.safetensors": keys[half:]}
    weight_map = {}
    for fname, ks in shards.items():
        save_file({k: t[k].contiguous() for k in ks}, os.path.join(args.dst, fname))
        weight_map.update({k: fname for k in ks})
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    if args.tokenizer:
        shutil.copy(args.tokenizer, os.path.join(args.dst, "tokenizer.json"))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
