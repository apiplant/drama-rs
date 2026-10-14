#!/usr/bin/env python3
"""Convert the Fish codec (codec.pth) into codec.safetensors for drama-rs.

Weight norm is folded, convolutions are re-laid out for time-major im2col GEMMs:
  conv   [Cout, Cin, k]          -> W [Cout, k*Cin]        (column j*Cin + ci)
  tconv  [Cin, Cout, 2s] stride s -> W [s*Cout, 2*Cin]      (row r*Cout + co; cols x[t] | x[t-1])
  tconv  [Cin, Cout, s]  stride s -> W [s*Cout, Cin]
GEMM weights are f16; everything else is f32.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "codec_py"))
import build_codec  # noqa: E402


def conv_w(conv: torch.nn.Conv1d) -> torch.Tensor:
    w = conv.weight.detach().float()  # [Cout, Cin, k]
    return w.permute(0, 2, 1).reshape(w.shape[0], -1).contiguous()


def tconv_w(conv: torch.nn.ConvTranspose1d, stride: int) -> torch.Tensor:
    w = conv.weight.detach().float()  # [Cin, Cout, k]
    cin, cout, k = w.shape
    if k == 2 * stride:
        cur = w[:, :, :stride]  # tap r applies to x[t]
        prev = w[:, :, stride:]  # tap r + s applies to x[t-1]
        rows = torch.cat([cur, prev], 0)  # [2Cin, Cout, s]
        return rows.permute(2, 1, 0).reshape(stride * cout, 2 * cin).contiguous()
    assert k == stride
    return w.permute(2, 1, 0).reshape(stride * cout, cin).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("codec_pth")
    ap.add_argument("out")
    args = ap.parse_args()
    codec = build_codec.build_codec(args.codec_pth, "cpu")
    codec.fold_weight_norm()
    for m in codec.modules():  # any remaining old-style weight norm
        if hasattr(m, "weight_g"):
            torch.nn.utils.remove_weight_norm(m)
    q = codec.quantizer
    t: dict[str, torch.Tensor] = {}
    h = lambda x: x.detach().to(torch.float16).contiguous()
    f = lambda x: x.detach().float().contiguous()

    # Per-code output vectors: out_proj(codebook[c]) + bias, semantic first.
    tables = []
    for vq in [q.semantic_quantizer.quantizers[0], *q.quantizer.quantizers]:
        cbk = vq.codebook.weight.detach().float()  # [n, 8]
        w = vq.out_proj.weight.detach().float().squeeze(-1)  # [1024, 8]
        tables.append(cbk @ w.T + vq.out_proj.bias.detach().float())
    t["rvq.table"] = torch.cat(tables, 0).contiguous()

    def transformer(prefix: str, tr) -> None:
        for j, layer in enumerate(tr.layers):
            p = f"{prefix}.l{j}."
            t[p + "attn_norm"] = f(layer.attention_norm.weight)
            t[p + "ffn_norm"] = f(layer.ffn_norm.weight)
            t[p + "attn_scale"] = f(layer.attention_layer_scale.gamma)
            t[p + "ffn_scale"] = f(layer.ffn_layer_scale.gamma)
            t[p + "wqkv"] = h(layer.attention.wqkv.weight)
            t[p + "wo"] = h(layer.attention.wo.weight)
            t[p + "w13"] = h(torch.cat([layer.feed_forward.w1.weight, layer.feed_forward.w3.weight], 0))
            t[p + "w2"] = h(layer.feed_forward.w2.weight)
        t[prefix + ".norm"] = f(tr.norm.weight)

    def convnext(prefix: str, blk) -> None:
        t[prefix + ".dw_w"] = f(blk.dwconv.conv.weight.squeeze(1))  # [C, 7]
        t[prefix + ".dw_b"] = f(blk.dwconv.conv.bias)
        t[prefix + ".ln_w"] = f(blk.norm.weight)
        t[prefix + ".ln_b"] = f(blk.norm.bias)
        t[prefix + ".pw1_w"] = h(blk.pwconv1.weight)
        t[prefix + ".pw1_b"] = f(blk.pwconv1.bias)
        t[prefix + ".pw2_w"] = h(blk.pwconv2.weight)
        t[prefix + ".pw2_b"] = f(blk.pwconv2.bias)
        t[prefix + ".gamma"] = f(blk.gamma)

    transformer("post", q.post_module)
    transformer("pre", q.pre_module)
    for i, (tconv, cnx) in enumerate(q.upsample):
        t[f"up{i}.w"] = h(tconv_w(tconv.conv, tconv.stride))
        t[f"up{i}.b"] = f(tconv.conv.bias)
        convnext(f"up{i}.cnx", cnx)

    dec = codec.decoder.model
    t["dec.in.w"] = h(conv_w(dec[0].conv))
    t["dec.in.b"] = f(dec[0].conv.bias)
    for b in range(4):
        blk = dec[1 + b].block
        p = f"dec.b{b}."
        t[p + "alpha"] = f(blk[0].alpha.reshape(-1))
        t[p + "up.w"] = h(tconv_w(blk[1].conv, blk[1].stride))
        t[p + "up.b"] = f(blk[1].conv.bias)
        for u in range(3):
            ru = blk[2 + u].block
            t[p + f"ru{u}.a1"] = f(ru[0].alpha.reshape(-1))
            t[p + f"ru{u}.c1.w"] = h(conv_w(ru[1].conv))
            t[p + f"ru{u}.c1.b"] = f(ru[1].conv.bias)
            t[p + f"ru{u}.a2"] = f(ru[2].alpha.reshape(-1))
            t[p + f"ru{u}.c2.w"] = h(conv_w(ru[3].conv))
            t[p + f"ru{u}.c2.b"] = f(ru[3].conv.bias)
    # ---- encoder side (reference audio -> codes)
    enc = codec.encoder.block
    t["enc.in.w"] = h(conv_w(enc[0].conv))
    t["enc.in.b"] = f(enc[0].conv.bias)
    for b in range(4):
        blk = enc[1 + b].block
        p = f"enc.b{b}."
        for u in range(3):
            ru = blk[u].block
            t[p + f"ru{u}.a1"] = f(ru[0].alpha.reshape(-1))
            t[p + f"ru{u}.c1.w"] = h(conv_w(ru[1].conv))
            t[p + f"ru{u}.c1.b"] = f(ru[1].conv.bias)
            t[p + f"ru{u}.a2"] = f(ru[2].alpha.reshape(-1))
            t[p + f"ru{u}.c2.w"] = h(conv_w(ru[3].conv))
            t[p + f"ru{u}.c2.b"] = f(ru[3].conv.bias)
        t[p + "alpha"] = f(blk[3].alpha.reshape(-1))
        t[p + "down.w"] = h(conv_w(blk[4].conv))
        t[p + "down.b"] = f(blk[4].conv.bias)
        if len(blk) > 5 and hasattr(blk[5], "layers"):
            transformer(f"enc.b{b}.tr", blk[5])
    t["enc.out.alpha"] = f(enc[5].alpha.reshape(-1))
    t["enc.out.w"] = h(conv_w(enc[6].conv))
    t["enc.out.b"] = f(enc[6].conv.bias)
    for i, (conv, cnx) in enumerate(q.downsample):
        t[f"down{i}.w"] = h(conv_w(conv.conv))
        t[f"down{i}.b"] = f(conv.conv.bias)
        convnext(f"down{i}.cnx", cnx)
    for i, vq in enumerate([q.semantic_quantizer.quantizers[0], *q.quantizer.quantizers]):
        t[f"vq{i}.in_w"] = f(vq.in_proj.weight.squeeze(-1))  # [8, 1024]
        t[f"vq{i}.in_b"] = f(vq.in_proj.bias)
        t[f"vq{i}.codebook_n"] = f(torch.nn.functional.normalize(vq.codebook.weight.float()))

    t["dec.out.alpha"] = f(dec[5].alpha.reshape(-1))
    t["dec.out.w"] = f(conv_w(dec[6].conv))  # [1, 7*96]
    t["dec.out.b"] = f(dec[6].conv.bias)
    save_file(t, args.out, metadata={"format": "drama-rs-codec-v1"})
    print(f"wrote {args.out}: {sum(v.numel() * v.element_size() for v in t.values()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
