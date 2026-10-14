#!/usr/bin/env python3
"""Write a full-size converted model with random weights (for speed / memory tests only)."""

from __future__ import annotations

import json
import os
import sys

import torch
from safetensors.torch import save_file


def q8(n: int, k: int) -> dict[str, torch.Tensor]:
    return {
        "q8": torch.randint(-127, 128, (n, k), dtype=torch.int8),
        "s": torch.full((n, k // 128), 0.02 / 127 / (k / 2048) ** 0.5, dtype=torch.float16),
    }


def q4(e: int, n: int, k: int) -> dict[str, torch.Tensor]:
    sm = torch.empty(e, n, k // 64, 2, dtype=torch.float16)
    sm[..., 0] = 0.004
    sm[..., 1] = -0.03
    return {"q4": torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8), "sm": sm}


def put(out, name, d):
    for k, v in d.items():
        out[f"{name}.{k}"] = v


def main() -> None:
    cfg = json.load(open(sys.argv[1]))
    dst = sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    tc, dc = cfg["text_config"], cfg["audio_decoder_config"]
    H = tc["hidden_size"]
    ones = lambda n: torch.ones(n)
    g = {
        "embed_tokens": (torch.randn(tc["vocab_size"], H) * 0.02).to(torch.bfloat16),
        "lm_head_cand": (torch.randn(dc["vocab_size"] + 1, H) * 0.02).to(torch.bfloat16),
        "final_norm": ones(H),
    }
    save_file(g, os.path.join(dst, "globals.safetensors"))
    nk, nv, dk, dv = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    E, mi, si = tc["num_experts"], tc["moe_intermediate_size"], tc["shared_expert_intermediate_size"]
    for i in range(tc["num_hidden_layers"]):
        o = {"input_norm": ones(H), "post_norm": ones(H)}
        if tc["layer_types"][i] == "linear_attention":
            put(o, "la.in_proj", q8(2 * nk * dk + nv * dv + nv * dv + 2 * nv, H))
            put(o, "la.out_proj", q8(H, nv * dv))
            o["la.conv_w"] = torch.randn(2 * nk * dk + nv * dv, 4) * 0.3
            o["la.A_log"] = torch.zeros(nv)
            o["la.dt_bias"] = torch.zeros(nv)
            o["la.norm"] = ones(dv)
        else:
            h, hkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
            put(o, "attn.qkv", q8(2 * h * hd + 2 * hkv * hd, H))
            put(o, "attn.o", q8(H, h * hd))
            o["attn.q_norm"] = ones(hd)
            o["attn.k_norm"] = ones(hd)
        o["moe.router"] = (torch.randn(E + 1, H) * 0.05).to(torch.bfloat16)
        put(o, "moe.shared_gate_up", q8(2 * si, H))
        put(o, "moe.shared_down", q8(H, si))
        put(o, "moe.gate_up", q4(E, 2 * mi, H))
        put(o, "moe.down", q4(E, H, mi))
        save_file(o, os.path.join(dst, f"layer{i:02d}.safetensors"))
        print("layer", i, flush=True)
    t = {"codebook_emb": (torch.randn(dc["vocab_size"] * dc["num_codebooks"], H) * 0.1).to(torch.bfloat16)}
    D, Hh, Hkv, hd, I = dc["dim"], dc["n_head"], dc["n_local_heads"], dc["head_dim"], dc["intermediate_size"]
    for j in range(dc["shared_enrichment_layers"] + dc["num_codebooks"] - 1):
        p = f"t{j}."
        t[p + "tok_emb"] = (torch.randn(dc["vocab_size"], D) * 0.1).to(torch.bfloat16)
        for n in ("enorm", "hnorm", "final_norm", "input_norm", "post_norm"):
            t[p + n] = ones(D)
        t[p + "q_norm"] = ones(hd)
        t[p + "k_norm"] = ones(hd)
        put(t, p + "eh_proj", q8(D, 2 * D))
        put(t, p + "qkv", q8(2 * Hh * hd + 2 * Hkv * hd, D))
        put(t, p + "o", q8(D, Hh * hd))
        put(t, p + "gate_up", q8(2 * I, D))
        put(t, p + "down", q8(D, I))
    put(t, "heads", q8((dc["num_codebooks"] - 1) * dc["vocab_size"], D))
    save_file(t, os.path.join(dst, "tower.safetensors"))
    json.dump({"format": "drama-rs-v1", "q4_group": 64, "q8_group": 128, "config": cfg}, open(os.path.join(dst, "drama.json"), "w"))
    print("done")


if __name__ == "__main__":
    main()
