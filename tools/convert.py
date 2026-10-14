#!/usr/bin/env python3
"""Convert the fishaudio/drama-3 checkpoint into the drama-rs quantized layout.

Routed experts:   int4, asymmetric, group 64 along K   (q4: u8 [E, N, K/2], sm: f16 [E, N, K/64, 2])
Dense linears:    int8, symmetric, group 128 along K   (q8: i8 [N, K],      s:  f16 [N, K/128])
Norm weights:     f32, with the zero-centred "+1" folded in where the model uses it
Everything else:  bf16 / f32 as noted

One output file per backbone layer plus globals.safetensors and tower.safetensors, so
the conversion never holds more than one layer in memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_BLOCK = 128
Q4_GROUP = 64
Q8_GROUP = 128


# ----------------------------------------------------------------------------- quantizers


def quant_q8(w: torch.Tensor) -> dict[str, torch.Tensor]:
    """Symmetric int8, one f16 scale per Q8_GROUP inputs of a row."""
    n, k = w.shape
    assert k % Q8_GROUP == 0, (n, k)
    g = w.float().reshape(n, k // Q8_GROUP, Q8_GROUP)
    amax = g.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / 127.0).half().float().clamp(min=1e-12)
    q = torch.round(g / scale).clamp(-127, 127).to(torch.int8)
    return {"q8": q.reshape(n, k).contiguous(), "s": scale.squeeze(-1).half().contiguous()}


def quant_q4_groups(g: torch.Tensor, steps: int = 24) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric 4-bit over the last dim of g [..., G] with a clipping search.

    Returns (q uint8 [..., G], scale f16 [...], min f16 [...]) with w ~= q * scale + min.
    """
    gmin = g.amin(-1, keepdim=True)
    gmax = g.amax(-1, keepdim=True)
    best_err = None
    best = None
    center = (gmin + gmax) * 0.5
    # Shrink the range from both sides towards its centre and keep the lowest MSE.
    for i in range(steps):
        shrink = 1.0 - 0.5 * i / steps
        lo_c = center + (gmin - center) * shrink
        hi_c = center + (gmax - center) * shrink
        scale = ((hi_c - lo_c) / 15.0).clamp(min=1e-10)
        scale_h = scale.half().float().clamp(min=1e-10)
        lo_h = lo_c.half().float()
        q = torch.round((g - lo_h) / scale_h).clamp(0, 15)
        # Refit min/scale by least squares for the chosen code assignment.
        deq = q * scale_h + lo_h
        err = (deq - g).pow(2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best = err, (q, scale_h, lo_h)
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best = tuple(torch.where(better, new, old) for new, old in zip((q, scale_h, lo_h), best))
    q, scale_h, lo_h = best
    # One least-squares refinement of (scale, min) given the codes.
    n = g.shape[-1]
    sq = q.sum(-1, keepdim=True)
    sqq = (q * q).sum(-1, keepdim=True)
    sw = g.sum(-1, keepdim=True)
    sqw = (q * g).sum(-1, keepdim=True)
    det = n * sqq - sq * sq
    ok = det.abs() > 1e-6
    ls_scale = torch.where(ok, (n * sqw - sq * sw) / torch.where(ok, det, 1.0), scale_h)
    ls_min = torch.where(ok, (sw - ls_scale * sq) / n, lo_h)
    ls_scale_h = ls_scale.half().float()
    ls_min_h = ls_min.half().float()
    ls_err = ((q * ls_scale_h + ls_min_h) - g).pow(2).sum(-1, keepdim=True)
    use_ls = (ls_err < best_err) & ok & (ls_scale_h > 0)
    scale_h = torch.where(use_ls, ls_scale_h, scale_h)
    lo_h = torch.where(use_ls, ls_min_h, lo_h)
    return q.to(torch.uint8), scale_h.squeeze(-1).half(), lo_h.squeeze(-1).half()


def quant_q4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w [E, N, K] -> packed u8 [E, N, K/2] (low nibble = even k), f16 [E, N, K/64, 2]."""
    e, n, k = w.shape
    assert k % Q4_GROUP == 0
    g = w.float().reshape(e, n, k // Q4_GROUP, Q4_GROUP)
    q, scale, mn = quant_q4_groups(g)
    q = q.reshape(e, n, k)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8).contiguous()
    sm = torch.stack((scale, mn), dim=-1).contiguous()
    return packed, sm


def dequant_q4(packed: torch.Tensor, sm: torch.Tensor) -> torch.Tensor:
    lo = (packed & 0xF).float()
    hi = (packed >> 4).float()
    q = torch.stack((lo, hi), dim=-1).flatten(-2)
    *lead, k = q.shape
    q = q.reshape(*lead, k // Q4_GROUP, Q4_GROUP)
    return (q * sm[..., 0:1].float() + sm[..., 1:2].float()).reshape(*lead, k)


def dequant_q8(q8: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n, k = q8.shape
    return (q8.float().reshape(n, k // Q8_GROUP, Q8_GROUP) * s.float().unsqueeze(-1)).reshape(n, k)


# ----------------------------------------------------------------------------- checkpoint access


class Checkpoint:
    def __init__(self, path: str, device: str) -> None:
        self.path = path
        self.device = device
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self.handles: dict[str, object] = {}

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(self.path, shard), framework="pt", device="cpu")
        return self.handles[shard].get_tensor(name).to(self.device)

    def get_dequant(self, name: str) -> torch.Tensor:
        """A weight in f32; block-FP8 weights are multiplied out by their scale_inv."""
        w = self.get(name)
        if w.dtype == torch.float8_e4m3fn:
            s = self.get(name + "_scale_inv").float()
            n, k = w.shape
            s = s.repeat_interleave(FP8_BLOCK, 0)[:n].repeat_interleave(FP8_BLOCK, 1)[:, :k]
            return w.float() * s
        return w.float()


def norm_plus_one(w: torch.Tensor) -> torch.Tensor:
    return (w.float() + 1.0).contiguous()


def put_q8(out: dict[str, torch.Tensor], name: str, w: torch.Tensor) -> None:
    for suffix, t in quant_q8(w).items():
        out[f"{name}.{suffix}"] = t.cpu()


# ----------------------------------------------------------------------------- converters


def convert_globals(ck: Checkpoint, cfg: dict) -> dict[str, torch.Tensor]:
    tc = cfg["text_config"]
    out: dict[str, torch.Tensor] = {}
    emb = ck.get("model.language_model.embed_tokens.weight")
    out["embed_tokens"] = emb.to(torch.bfloat16).cpu().contiguous()
    lm_head = ck.get("lm_head.weight")
    eos = cfg["eos_token_id"]
    sem0, sem1 = cfg["semantic_start_token_id"], cfg["semantic_end_token_id"]
    candidates = sorted({eos, *range(sem0, sem1 + 1)})
    idx = torch.tensor(candidates, device=lm_head.device)
    out["lm_head_cand"] = lm_head.index_select(0, idx).to(torch.bfloat16).cpu().contiguous()
    out["lm_head_cand_ids"] = torch.tensor(candidates, dtype=torch.int32)
    out["final_norm"] = norm_plus_one(ck.get("model.language_model.norm.weight")).cpu()
    assert lm_head.shape[0] == tc["vocab_size"]
    return out


def convert_layer(ck: Checkpoint, cfg: dict, i: int) -> dict[str, torch.Tensor]:
    tc = cfg["text_config"]
    p = f"model.language_model.layers.{i}."
    out: dict[str, torch.Tensor] = {}
    out["input_norm"] = norm_plus_one(ck.get(p + "input_layernorm.weight")).cpu()
    out["post_norm"] = norm_plus_one(ck.get(p + "post_attention_layernorm.weight")).cpu()
    layer_type = tc["layer_types"][i]
    if layer_type == "linear_attention":
        la = p + "linear_attn."
        # One fused input projection: [qkv (conv channels) | z | b | a].
        w = torch.cat(
            [
                ck.get_dequant(la + "in_proj_qkv.weight"),
                ck.get_dequant(la + "in_proj_z.weight"),
                ck.get_dequant(la + "in_proj_b.weight"),
                ck.get_dequant(la + "in_proj_a.weight"),
            ],
            dim=0,
        )
        put_q8(out, "la.in_proj", w)
        put_q8(out, "la.out_proj", ck.get_dequant(la + "out_proj.weight"))
        conv = ck.get(la + "conv1d.weight").float()  # [C, 1, 4]
        out["la.conv_w"] = conv.reshape(conv.shape[0], conv.shape[-1]).contiguous().cpu()
        out["la.A_log"] = ck.get(la + "A_log").float().cpu()
        out["la.dt_bias"] = ck.get(la + "dt_bias").float().cpu()
        # RMSNormGated multiplies by the plain weight (no +1).
        out["la.norm"] = ck.get(la + "norm.weight").float().contiguous().cpu()
    else:
        sa = p + "self_attn."
        heads, hd = tc["num_attention_heads"], tc["head_dim"]
        q = ck.get_dequant(sa + "q_proj.weight")  # per head [query | gate]
        q = q.reshape(heads, 2, hd, -1)
        w = torch.cat(
            [
                q[:, 0].reshape(heads * hd, -1),
                q[:, 1].reshape(heads * hd, -1),
                ck.get_dequant(sa + "k_proj.weight"),
                ck.get_dequant(sa + "v_proj.weight"),
            ],
            dim=0,
        )  # [query | gate | key | value], the tower's layout
        put_q8(out, "attn.qkv", w)
        put_q8(out, "attn.o", ck.get_dequant(sa + "o_proj.weight"))
        out["attn.q_norm"] = norm_plus_one(ck.get(sa + "q_norm.weight")).cpu()
        out["attn.k_norm"] = norm_plus_one(ck.get(sa + "k_norm.weight")).cpu()

    mp = p + "mlp."
    # Router rows 0..E-1 plus the shared-expert gate as row E.
    router = torch.cat([ck.get_dequant(mp + "gate.weight"), ck.get_dequant(mp + "shared_expert_gate.weight")], 0)
    out["moe.router"] = router.to(torch.bfloat16).contiguous().cpu()
    put_q8(
        out,
        "moe.shared_gate_up",
        torch.cat([ck.get_dequant(mp + "shared_expert.gate_proj.weight"), ck.get_dequant(mp + "shared_expert.up_proj.weight")], 0),
    )
    put_q8(out, "moe.shared_down", ck.get_dequant(mp + "shared_expert.down_proj.weight"))

    num_experts = tc["num_experts"]
    chunk = 32
    gu_q, gu_s, d_q, d_s = [], [], [], []
    for e0 in range(0, num_experts, chunk):
        gate_up = torch.stack(
            [
                torch.cat(
                    [
                        ck.get_dequant(f"{mp}experts.{e}.gate_proj.weight"),
                        ck.get_dequant(f"{mp}experts.{e}.up_proj.weight"),
                    ],
                    0,
                )
                for e in range(e0, min(e0 + chunk, num_experts))
            ]
        )
        down = torch.stack(
            [ck.get_dequant(f"{mp}experts.{e}.down_proj.weight") for e in range(e0, min(e0 + chunk, num_experts))]
        )
        a, b = quant_q4(gate_up)
        gu_q.append(a.cpu())
        gu_s.append(b.cpu())
        a, b = quant_q4(down)
        d_q.append(a.cpu())
        d_s.append(b.cpu())
    out["moe.gate_up.q4"] = torch.cat(gu_q).contiguous()
    out["moe.gate_up.sm"] = torch.cat(gu_s).contiguous()
    out["moe.down.q4"] = torch.cat(d_q).contiguous()
    out["moe.down.sm"] = torch.cat(d_s).contiguous()
    return out


def convert_tower(ck: Checkpoint, cfg: dict) -> dict[str, torch.Tensor]:
    dc = cfg["audio_decoder_config"]
    out: dict[str, torch.Tensor] = {}
    out["codebook_emb"] = ck.get("audio_decoder.codebook_embeddings.weight").to(torch.bfloat16).cpu().contiguous()
    names = [f"audio_decoder_mtp_shared.layers.{j}." for j in range(dc["shared_enrichment_layers"])]
    names += [f"audio_decoder_mtp.layers.{j}." for j in range(dc["num_codebooks"] - 1)]
    for t, p in enumerate(names):
        o = f"t{t}."
        emb_name = p + ("semantic_embedding.weight" if "shared" in p else "codebook_embedding.weight")
        out[o + "tok_emb"] = ck.get(emb_name).to(torch.bfloat16).cpu().contiguous()
        out[o + "enorm"] = norm_plus_one(ck.get(p + "enorm.weight")).cpu()
        out[o + "hnorm"] = norm_plus_one(ck.get(p + "hnorm.weight")).cpu()
        out[o + "final_norm"] = norm_plus_one(ck.get(p + "final_layernorm.weight")).cpu()
        tl = p + "transformer_layer."
        out[o + "input_norm"] = norm_plus_one(ck.get(tl + "attention_norm.weight")).cpu()
        out[o + "post_norm"] = norm_plus_one(ck.get(tl + "ffn_norm.weight")).cpu()
        out[o + "q_norm"] = norm_plus_one(ck.get(tl + "attention.q_norm.weight")).cpu()
        out[o + "k_norm"] = norm_plus_one(ck.get(tl + "attention.k_norm.weight")).cpu()
        put_q8(out, o + "eh_proj", ck.get_dequant(p + "eh_proj.weight"))
        put_q8(out, o + "qkv", ck.get_dequant(tl + "attention.wqkv.weight"))  # [query | gate | key | value]
        put_q8(out, o + "o", ck.get_dequant(tl + "attention.wo.weight"))
        put_q8(
            out,
            o + "gate_up",
            torch.cat([ck.get_dequant(tl + "feed_forward.w1.weight"), ck.get_dequant(tl + "feed_forward.w3.weight")], 0),
        )
        put_q8(out, o + "down", ck.get_dequant(tl + "feed_forward.w2.weight"))
    heads = [ck.get_dequant(f"audio_decoder_mtp.heads.{j}.weight") for j in range(dc["num_codebooks"] - 1)]
    put_q8(out, "heads", torch.cat(heads, 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory of the original checkpoint")
    ap.add_argument("dst", help="output directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="comma list: globals,tower,layer:N")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    with open(os.path.join(args.src, "config.json")) as f:
        cfg = json.load(f)
    ck = Checkpoint(args.src, args.device)
    tc = cfg["text_config"]
    jobs: list[tuple[str, object]] = [("globals", None), ("tower", None)]
    jobs += [(f"layer:{i}", i) for i in range(tc["num_hidden_layers"])]
    if args.only:
        wanted = set(args.only.split(","))
        jobs = [j for j in jobs if j[0] in wanted]

    for name, arg in jobs:
        fname = "globals.safetensors" if name == "globals" else "tower.safetensors" if name == "tower" else f"layer{arg:02d}.safetensors"
        path = os.path.join(args.dst, fname)
        if os.path.exists(path):
            print(f"skip {fname} (exists)")
            continue
        t0 = time.time()
        with torch.no_grad():
            if name == "globals":
                tensors = convert_globals(ck, cfg)
            elif name == "tower":
                tensors = convert_tower(ck, cfg)
            else:
                tensors = convert_layer(ck, cfg, arg)
        save_file(tensors, path + ".tmp", metadata={"format": "drama-rs-v1"})
        os.replace(path + ".tmp", path)
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        print(f"{fname}: {size / 1e9:.3f} GB in {time.time() - t0:.1f}s", flush=True)
        if args.device == "cuda":
            torch.cuda.empty_cache()

    meta = {
        "format": "drama-rs-v1",
        "q4_group": Q4_GROUP,
        "q8_group": Q8_GROUP,
        "config": cfg,
    }
    with open(os.path.join(args.dst, "drama.json"), "w") as f:
        json.dump(meta, f, indent=1)
    for extra in ("tokenizer.json", "codec.pth"):
        src = os.path.join(args.src, extra)
        dst = os.path.join(args.dst, extra)
        if os.path.exists(src) and not os.path.exists(dst):
            os.link(src, dst) if os.stat(src).st_dev == os.stat(args.dst).st_dev else None


if __name__ == "__main__":
    main()
