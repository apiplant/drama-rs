#!/usr/bin/env python3
"""A plain PyTorch reference of Drama 3 over the drama-rs converted weights.

fp32 math throughout; it exists to check the Rust/CUDA engine, not to be fast.
Mirrors sglang-omni's fishaudio_drama3 model (Qwen3.5-MoE backbone + codebook tower).
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

Q4_GROUP = 64
Q8_GROUP = 128
RAS_WINDOW = 16


def dequant_q8(q8: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n, k = q8.shape
    return (q8.float().reshape(n, k // Q8_GROUP, Q8_GROUP) * s.float().unsqueeze(-1)).reshape(n, k)


def dequant_q4(packed: torch.Tensor, sm: torch.Tensor) -> torch.Tensor:
    lo = (packed & 0xF).float()
    hi = (packed >> 4).float()
    q = torch.stack((lo, hi), dim=-1).flatten(-2)
    *lead, k = q.shape
    q = q.reshape(*lead, k // Q4_GROUP, Q4_GROUP)
    return (q * sm[..., 0:1].float() + sm[..., 1:2].float()).reshape(*lead, k)


class Weights:
    """Converted tensors, loaded to the device on first use."""

    def __init__(self, path: str, device: str = "cuda") -> None:
        self.device = device
        self.files: dict[str, object] = {}
        self.where: dict[str, str] = {}
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".safetensors"):
                h = safe_open(os.path.join(path, fname), framework="pt", device="cpu")
                prefix = "" if fname in ("globals.safetensors", "tower.safetensors") else fname.split(".")[0] + "."
                self.files[fname] = h
                for k in h.keys():
                    self.where[prefix + k] = (fname, k)
        self.cache: dict[str, torch.Tensor] = {}

    def raw(self, name: str) -> torch.Tensor:
        if name not in self.cache:
            fname, key = self.where[name]
            self.cache[name] = self.files[fname].get_tensor(key).to(self.device)
        return self.cache[name]

    def f(self, name: str) -> torch.Tensor:
        return self.raw(name).float()

    def dense(self, name: str) -> torch.Tensor:
        """A q8 linear dequantized to f32 [N, K]."""
        return dequant_q8(self.raw(name + ".q8"), self.raw(name + ".s"))

    def expert(self, layer: str, which: str, e: int) -> torch.Tensor:
        return dequant_q4(self.raw(f"{layer}.moe.{which}.q4")[e], self.raw(f"{layer}.moe.{which}.sm")[e])


class OrigWeights:
    """The original fishaudio/drama-3 checkpoint behind the same interface as Weights:
    dense tensors cached on the GPU in bf16, FP8 experts dequantized on demand."""

    def __init__(self, path: str, device: str = "cuda") -> None:
        sys.path.insert(0, os.path.dirname(__file__))
        import convert  # noqa: E402

        self.convert = convert
        self.ck = convert.Checkpoint(path, "cpu")
        with open(os.path.join(path, "config.json")) as f:
            self.cfg = json.load(f)
        self.device = device
        self.cache: dict[str, torch.Tensor] = {}

    def _orig(self, name: str) -> torch.Tensor:
        return self.ck.get_dequant(name).to(self.device)

    def _build(self, name: str) -> torch.Tensor:
        """Converted-layout tensor `name` (dense, unquantized), mirroring convert.py."""
        tc = self.cfg["text_config"]
        if name == "embed_tokens":
            return self._orig("model.language_model.embed_tokens.weight")
        if name == "lm_head_cand":
            eos, s0, s1 = self.cfg["eos_token_id"], self.cfg["semantic_start_token_id"], self.cfg["semantic_end_token_id"]
            ids = torch.tensor(sorted({eos, *range(s0, s1 + 1)}))
            return self.ck.get("lm_head.weight").index_select(0, ids).to(self.device)
        if name == "final_norm":
            return self._orig("model.language_model.norm.weight") + 1
        if name == "codebook_emb":
            return self._orig("audio_decoder.codebook_embeddings.weight")
        if name.startswith("layer"):
            i = int(name[5:7])
            rest = name[8:]
            p = f"model.language_model.layers.{i}."
            la, sa, mp = p + "linear_attn.", p + "self_attn.", p + "mlp."
            table = {
                "input_norm": lambda: self._orig(p + "input_layernorm.weight") + 1,
                "post_norm": lambda: self._orig(p + "post_attention_layernorm.weight") + 1,
                "la.in_proj": lambda: torch.cat([self._orig(la + k + ".weight") for k in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]),
                "la.out_proj": lambda: self._orig(la + "out_proj.weight"),
                "la.conv_w": lambda: self._orig(la + "conv1d.weight").reshape(-1, tc["linear_conv_kernel_dim"]),
                "la.A_log": lambda: self._orig(la + "A_log"),
                "la.dt_bias": lambda: self._orig(la + "dt_bias"),
                "la.norm": lambda: self._orig(la + "norm.weight"),
                "attn.qkv": lambda: self._attn_qkv(sa),
                "attn.o": lambda: self._orig(sa + "o_proj.weight"),
                "attn.q_norm": lambda: self._orig(sa + "q_norm.weight") + 1,
                "attn.k_norm": lambda: self._orig(sa + "k_norm.weight") + 1,
                "moe.router": lambda: torch.cat([self._orig(mp + "gate.weight"), self._orig(mp + "shared_expert_gate.weight")]),
                "moe.shared_gate_up": lambda: torch.cat([self._orig(mp + "shared_expert.gate_proj.weight"), self._orig(mp + "shared_expert.up_proj.weight")]),
                "moe.shared_down": lambda: self._orig(mp + "shared_expert.down_proj.weight"),
            }
            return table[rest]()
        if name == "heads":
            dc = self.cfg["audio_decoder_config"]
            return torch.cat([self._orig(f"audio_decoder_mtp.heads.{j}.weight") for j in range(dc["num_codebooks"] - 1)])
        if name.startswith("t"):
            t, rest = name[1:].split(".", 1)
            t = int(t)
            dc = self.cfg["audio_decoder_config"]
            ns = dc["shared_enrichment_layers"]
            p = f"audio_decoder_mtp_shared.layers.{t}." if t < ns else f"audio_decoder_mtp.layers.{t - ns}."
            tl = p + "transformer_layer."
            table = {
                "tok_emb": lambda: self._orig(p + ("semantic_embedding.weight" if t < ns else "codebook_embedding.weight")),
                "enorm": lambda: self._orig(p + "enorm.weight") + 1,
                "hnorm": lambda: self._orig(p + "hnorm.weight") + 1,
                "final_norm": lambda: self._orig(p + "final_layernorm.weight") + 1,
                "input_norm": lambda: self._orig(tl + "attention_norm.weight") + 1,
                "post_norm": lambda: self._orig(tl + "ffn_norm.weight") + 1,
                "q_norm": lambda: self._orig(tl + "attention.q_norm.weight") + 1,
                "k_norm": lambda: self._orig(tl + "attention.k_norm.weight") + 1,
                "eh_proj": lambda: self._orig(p + "eh_proj.weight"),
                "qkv": lambda: self._orig(tl + "attention.wqkv.weight"),
                "o": lambda: self._orig(tl + "attention.wo.weight"),
                "gate_up": lambda: torch.cat([self._orig(tl + "feed_forward.w1.weight"), self._orig(tl + "feed_forward.w3.weight")]),
                "down": lambda: self._orig(tl + "feed_forward.w2.weight"),
            }
            return table[rest]()
        raise KeyError(name)

    def _attn_qkv(self, sa: str) -> torch.Tensor:
        tc = self.cfg["text_config"]
        h, hd = tc["num_attention_heads"], tc["head_dim"]
        q = self._orig(sa + "q_proj.weight").reshape(h, 2, hd, -1)
        return torch.cat([q[:, 0].reshape(h * hd, -1), q[:, 1].reshape(h * hd, -1), self._orig(sa + "k_proj.weight"), self._orig(sa + "v_proj.weight")])

    def _get(self, name: str) -> torch.Tensor:
        if name not in self.cache:
            t = self._build(name)
            # Big matrices stay bf16 (the checkpoint's own precision); small ones f32.
            self.cache[name] = t.to(torch.bfloat16) if t.numel() > 1 << 20 else t.float()
        return self.cache[name]

    def raw(self, name: str) -> torch.Tensor:
        return self._get(name)

    def f(self, name: str) -> torch.Tensor:
        return self._get(name).float()

    def dense(self, name: str) -> torch.Tensor:
        return self._get(name).float()

    def expert(self, layer: str, which: str, e: int) -> torch.Tensor:
        i = int(layer[5:7])
        mp = f"model.language_model.layers.{i}.mlp.experts.{e}."
        if which == "gate_up":
            return torch.cat([self._orig(mp + "gate_proj.weight"), self._orig(mp + "up_proj.weight")]).float()
        return self._orig(mp + "down_proj.weight").float()


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_neox(x: torch.Tensor, pos: torch.Tensor, rot: int, theta: float) -> torch.Tensor:
    """x [L, H, D]; rotate pairs (i, i + rot/2) of the first rot dims."""
    inv = 1.0 / theta ** (torch.arange(0, rot, 2, device=x.device, dtype=torch.float32) / rot)
    ang = pos.float()[:, None] * inv[None]
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    half = rot // 2
    x1, x2, rest = x[..., :half], x[..., half:rot], x[..., rot:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), -1)


def rope_interleaved(x: torch.Tensor, pos: torch.Tensor, rot: int, theta: float) -> torch.Tensor:
    """x [L, H, D]; rotate pairs (2i, 2i+1) of the first rot dims."""
    inv = 1.0 / theta ** (torch.arange(0, rot, 2, device=x.device, dtype=torch.float32) / rot)
    ang = pos.float()[:, None] * inv[None]
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    ev, od = x[..., 0:rot:2], x[..., 1:rot:2]
    r = torch.stack((ev * cos - od * sin, od * cos + ev * sin), -1).flatten(-2)
    return torch.cat((r, x[..., rot:]), -1)


def attention(q, k, v, q_pos, causal_len_offset=0):
    """q [Lq, H, D], k/v [Lk, Hkv, D]; query i sees keys 0..q_pos[i]."""
    h, hkv = q.shape[1], k.shape[1]
    rep = h // hkv
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(q.shape[-1])
    kpos = torch.arange(k.shape[0], device=q.device)
    mask = kpos[None, :] > q_pos[:, None]
    scores = scores.masked_fill(mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


class State:
    def __init__(self, model: "Drama3Ref", max_len: int) -> None:
        tc = model.tc
        dev = model.dev
        self.len = 0
        self.conv = {}
        self.rec = {}
        self.kv = {}
        self.tkv = {}
        self.ras: list[int] = []


class Drama3Ref:
    def __init__(self, path: str, device: str = "cuda", original: str | None = None) -> None:
        """path: converted model dir. original: if given, run the unquantized original
        checkpoint from that directory instead of the converted weights."""
        with open(os.path.join(path, "drama.json")) as f:
            self.meta = json.load(f)
        self.cfg = self.meta["config"]
        self.tc = self.cfg["text_config"]
        self.dc = self.cfg["audio_decoder_config"]
        self.w = OrigWeights(original, device) if original else Weights(path, device)
        self.dev = device
        self.eps = self.tc["rms_norm_eps"]
        self.deps = self.dc["norm_eps"]
        self.sem0 = self.cfg["semantic_start_token_id"]
        self.sem1 = self.cfg["semantic_end_token_id"]
        self.eos = self.cfg["eos_token_id"]
        self.cb = self.dc["vocab_size"]
        self.ncb = self.dc["num_codebooks"]
        self.n_tower = self.dc["shared_enrichment_layers"] + self.ncb - 1

    # ------------------------------------------------------------------ embeddings

    def is_sem(self, tok: torch.Tensor) -> torch.Tensor:
        return (tok >= self.sem0) & (tok <= self.sem1)

    def cb_sum(self, frames: torch.Tensor, first: int, end: int) -> torch.Tensor:
        table = self.w.f("codebook_emb")
        out = torch.zeros(frames.shape[0], table.shape[1], device=self.dev)
        for c in range(first, end):
            out += table[frames[:, 1 + c].clamp(0, self.cb - 1) + c * self.cb]
        return out

    def embed(self, frames: torch.Tensor) -> torch.Tensor:
        emb = self.w.f("embed_tokens")[frames[:, 0]]
        nb = self.dc["vq_backbone_codebooks"]
        mixed = (emb + self.cb_sum(frames, 1, 1 + nb)) * (1.0 / math.sqrt(1.0 + nb))
        return torch.where(self.is_sem(frames[:, 0])[:, None], mixed, emb)

    # ------------------------------------------------------------------ backbone

    def linear_attn(self, i: int, x: torch.Tensor, st: State) -> torch.Tensor:
        L = x.shape[0]
        p = f"layer{i:02d}.la."
        tc = self.tc
        nk, nv = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
        dk, dv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
        kd, vd = nk * dk, nv * dv
        proj = x @ self.w.dense(p + "in_proj").T
        qkv, z, b, a = proj.split([2 * kd + vd, vd, nv, nv], -1)
        conv_w = self.w.f(p + "conv_w")  # [C, 4]
        ksz = conv_w.shape[1]
        prev = st.conv.get(i, torch.zeros(ksz - 1, qkv.shape[1], device=self.dev))
        seq = torch.cat([prev, qkv], 0)
        st.conv[i] = seq[-(ksz - 1):].clone()
        out = torch.zeros_like(qkv)
        for j in range(ksz):
            out += seq[j : j + L] * conv_w[:, j]
        out = F.silu(out)
        q, k, v = out.split([kd, kd, vd], -1)
        q = q.reshape(L, nk, dk).repeat_interleave(nv // nk, 1)
        k = k.reshape(L, nk, dk).repeat_interleave(nv // nk, 1)
        v = v.reshape(L, nv, dv)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        beta = b.sigmoid()
        g = -self.w.f(p + "A_log").exp() * F.softplus(a + self.w.f(p + "dt_bias"))
        S = st.rec.get(i, torch.zeros(nv, dk, dv, device=self.dev))
        o = torch.zeros(L, nv, dv, device=self.dev)
        for t in range(L):
            S = S * g[t].exp()[:, None, None]
            kv_mem = (S * k[t][:, :, None]).sum(1)
            delta = (v[t] - kv_mem) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            o[t] = (S * q[t][:, :, None]).sum(1)
        st.rec[i] = S
        o = rms(o, self.w.f(p + "norm"), self.eps) * F.silu(z.reshape(L, nv, dv))
        return o.reshape(L, vd) @ self.w.dense(p + "out_proj").T

    def full_attn(self, i: int, x: torch.Tensor, pos: torch.Tensor, st: State) -> torch.Tensor:
        L = x.shape[0]
        p = f"layer{i:02d}.attn."
        tc = self.tc
        h, hkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
        qkv = x @ self.w.dense(p + "qkv").T
        q, gate, k, v = qkv.split([h * hd, h * hd, hkv * hd, hkv * hd], -1)
        q = rms(q.reshape(L, h, hd), self.w.f(p + "q_norm"), self.eps)
        k = rms(k.reshape(L, hkv, hd), self.w.f(p + "k_norm"), self.eps)
        rot = int(hd * tc["rope_parameters"]["partial_rotary_factor"])
        theta = tc["rope_parameters"]["rope_theta"]
        q = rope_neox(q, pos, rot, theta)
        k = rope_neox(k, pos, rot, theta)
        v = v.reshape(L, hkv, hd)
        kc, vc = st.kv.get(i, (torch.zeros(0, hkv, hd, device=self.dev),) * 2)
        kc, vc = torch.cat([kc, k]), torch.cat([vc, v])
        st.kv[i] = (kc, vc)
        o = attention(q, kc, vc, pos)
        o = o.reshape(L, h * hd) * gate.sigmoid()
        return o @ self.w.dense(p + "o").T

    def moe(self, i: int, x: torch.Tensor) -> torch.Tensor:
        p = f"layer{i:02d}"
        tc = self.tc
        E, topk = tc["num_experts"], tc["num_experts_per_tok"]
        logits = x @ self.w.f(p + ".moe.router").T
        router, shared_gate = logits[:, :E], logits[:, E]
        probs = router.softmax(-1)
        topv, topi = probs.topk(topk, -1)
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        inter = tc["moe_intermediate_size"]
        for e in topi.unique().tolist():
            rows, slot = (topi == e).nonzero(as_tuple=True)
            gu = x[rows] @ self.w.expert(p, "gate_up", e).T
            act = F.silu(gu[:, :inter]) * gu[:, inter:]
            y = act @ self.w.expert(p, "down", e).T
            out.index_add_(0, rows, y * topv[rows, slot][:, None])
        sgu = x @ self.w.dense(p + ".moe.shared_gate_up").T
        sh = tc["shared_expert_intermediate_size"]
        s = (F.silu(sgu[:, :sh]) * sgu[:, sh:]) @ self.w.dense(p + ".moe.shared_down").T
        return out + shared_gate.sigmoid()[:, None] * s

    def backbone(self, frames: torch.Tensor, pos: torch.Tensor, st: State) -> torch.Tensor:
        x = self.embed(frames)
        for i in range(self.tc["num_hidden_layers"]):
            p = f"layer{i:02d}."
            hn = rms(x, self.w.f(p + "input_norm"), self.eps)
            if self.tc["layer_types"][i] == "linear_attention":
                x = x + self.linear_attn(i, hn, st)
            else:
                x = x + self.full_attn(i, hn, pos, st)
            x = x + self.moe(i, rms(x, self.w.f(p + "post_norm"), self.eps))
        return rms(x, self.w.f("final_norm"), self.eps)

    def main_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Logits over the candidates [eos, semantic 0..cb-1]."""
        return h @ self.w.f("lm_head_cand").T

    # ------------------------------------------------------------------ tower

    def tower_layer(self, t, h, code, label_sem, shared_in, pos, st: State):
        L = h.shape[0]
        p = f"t{t}."
        dc = self.dc
        eps = self.deps
        e = self.w.f(p + "tok_emb")[code.clamp(0, self.cb - 1)]
        e = torch.where(label_sem[:, None], e, torch.zeros_like(e))
        if shared_in is not None:
            e = e + shared_in
        mixed = torch.cat([rms(e, self.w.f(p + "enorm"), eps), rms(h, self.w.f(p + "hnorm"), eps)], -1)
        mixed = mixed @ self.w.dense(p + "eh_proj").T
        a_in = rms(mixed, self.w.f(p + "input_norm"), eps)
        H, Hkv, hd = dc["n_head"], dc["n_local_heads"], dc["head_dim"]
        qkv = a_in @ self.w.dense(p + "qkv").T
        q, gate, k, v = qkv.split([H * hd, H * hd, Hkv * hd, Hkv * hd], -1)
        q = rms(q.reshape(L, H, hd), self.w.f(p + "q_norm"), eps)
        k = rms(k.reshape(L, Hkv, hd), self.w.f(p + "k_norm"), eps)
        rot = int(hd * dc["rotary_percent"])
        q = rope_interleaved(q, pos, rot, dc["rope_base"])
        k = rope_interleaved(k, pos, rot, dc["rope_base"])
        v = v.reshape(L, Hkv, hd)
        kc, vc = st.tkv.get(t, (torch.zeros(0, Hkv, hd, device=self.dev),) * 2)
        kc, vc = torch.cat([kc, k]), torch.cat([vc, v])
        st.tkv[t] = (kc, vc)
        o = attention(q, kc, vc, pos).reshape(L, H * hd) * gate.sigmoid()
        r = o @ self.w.dense(p + "o").T + mixed
        f_in = rms(r, self.w.f(p + "post_norm"), eps)
        gu = f_in @ self.w.dense(p + "gate_up").T
        inter = dc["intermediate_size"]
        m = (F.silu(gu[:, :inter]) * gu[:, inter:]) @ self.w.dense(p + "down").T
        return rms(m + r, self.w.f(p + "final_norm"), eps)

    def shared_input(self, frames: torch.Tensor) -> torch.Tensor:
        nb = self.dc["vq_backbone_codebooks"]
        s = self.cb_sum(frames, 1 + nb, self.ncb)
        return torch.where(self.is_sem(frames[:, 0])[:, None], s, torch.zeros_like(s))

    def head_logits(self, r: int, h: torch.Tensor) -> torch.Tensor:
        heads = self.w.dense("heads")
        return h @ heads[r * self.cb : (r + 1) * self.cb].T

    # ------------------------------------------------------------------ sampling

    @staticmethod
    def sample(logits, temperature, top_k, top_p, gen):
        """sglang-omni's sample_top_k_top_p for one row."""
        scaled = logits.float() / max(temperature, 1e-5)
        sl, si = torch.sort(scaled, descending=True)
        cum = torch.cumsum(sl.softmax(-1), -1)
        ranks = torch.arange(sl.shape[0], device=sl.device)
        drop = (cum > top_p) | (ranks >= top_k)
        drop[0] = False
        probs = sl.masked_fill(drop, float("-inf")).softmax(-1)
        if temperature <= 0:
            return int(si[0])
        choice = torch.multinomial(probs, 1, generator=gen)
        return int(si[choice])

    # ------------------------------------------------------------------ driver

    def new_state(self) -> State:
        return State(self, 0)

    def prefill(self, frames: torch.Tensor, st: State):
        """Run the prompt rows; returns the last row's final hidden state."""
        L = frames.shape[0]
        pos = torch.arange(st.len, st.len + L, device=self.dev)
        h = self.backbone(frames, pos, st)
        nxt = torch.roll(frames, -1, 0)
        nxt[-1] = 0
        nxt[-1, 0] = self.eos
        self.tower_all(h[:-1], frames[:-1], nxt[:-1], pos[:-1], st)
        st.len += L
        st.ras = [int(t) for t in frames[-RAS_WINDOW:, 0].tolist()]
        return h[-1:], pos[-1:], frames[-1:]

    def tower_all(self, h, in_frames, next_frames, pos, st):
        """Teacher-forced tower pass that fills the tower caches for these rows."""
        if h.shape[0] == 0:
            for t in range(self.n_tower):
                st.tkv.setdefault(t, (torch.zeros(0, self.dc["n_local_heads"], self.dc["head_dim"], device=self.dev),) * 2)
            return
        lab = self.is_sem(next_frames[:, 0])
        shared = self.shared_input(in_frames)
        ns = self.dc["shared_enrichment_layers"]
        for t in range(ns):
            h = self.tower_layer(t, h, next_frames[:, 1], lab, shared if t == 0 else None, pos, st)
        for r in range(self.ncb - 1):
            h = self.tower_layer(ns + r, h, next_frames[:, 1 + r], lab, None, pos, st)

    def decode_frame(self, h, pos, in_frame, st: State, temperature, top_k, top_p, gen, forced=None, trace=None):
        """Sample the next frame from the last hidden state; fills tower caches at pos."""
        cand = torch.tensor([self.eos, *range(self.sem0, self.sem1 + 1)], device=self.dev)
        logits = self.main_logits(h)[0]
        if trace is not None:
            trace["main_logits"] = logits.clone()
        if forced is None:
            first = int(cand[self.sample(logits, temperature, top_k, top_p, gen)])
            ras = int(cand[self.sample(logits, 1.0, 30, 0.95, gen)])
            window = st.ras[-RAS_WINDOW:]
            if self.sem0 <= first <= self.sem1 and first in window:
                main = ras
            else:
                main = first
        else:
            main = int(forced[0])
        code0 = min(max(main - self.sem0, 0), self.cb - 1)
        lab = self.is_sem(torch.tensor([main], device=self.dev))
        shared = self.shared_input(in_frame)
        ns = self.dc["shared_enrichment_layers"]
        codes = [code0]
        x = h
        for t in range(ns):
            x = self.tower_layer(t, x, torch.tensor([code0], device=self.dev), lab, shared if t == 0 else None, pos, st)
        for r in range(self.ncb - 1):
            x = self.tower_layer(ns + r, x, torch.tensor([codes[-1]], device=self.dev), lab, None, pos, st)
            hl = self.head_logits(r, x)[0]
            if trace is not None:
                trace[f"head{r}"] = hl.clone()
            if forced is None:
                codes.append(self.sample(hl, temperature, top_k, top_p, gen))
            else:
                codes.append(int(forced[2 + r]))
        st.ras = (st.ras + [main])[-RAS_WINDOW:]
        return [main, *codes]

    def step(self, frame: list[int], st: State):
        """Feed one sampled frame through the backbone; returns its hidden state."""
        f = torch.tensor([frame], device=self.dev)
        pos = torch.tensor([st.len], device=self.dev)
        h = self.backbone(f, pos, st)
        st.len += 1
        return h, pos, f
