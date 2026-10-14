#!/usr/bin/env python3
"""Encode a clip with the original codec, decode with it and with drama-rs, compare."""
import os
import subprocess
import sys

import numpy as np
import wave

import torch
import torchaudio


def read_wav(path):
    with wave.open(path) as w:
        sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = np.frombuffer(w.readframes(n), dtype={2: "<i2", 4: "<i4"}[width]).reshape(-1, ch)
    return torch.from_numpy(raw.astype(np.float32).T / float(2 ** (8 * width - 1))), sr


def write_wav(path, audio, sr=44100):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).round().astype("<i2").tobytes())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref", "codec_py"))
import build_codec  # noqa: E402

codec_pth, wav, work = sys.argv[1], sys.argv[2], sys.argv[3]
codec_st = sys.argv[4] if len(sys.argv) > 4 else None
os.makedirs(work, exist_ok=True)
codec = build_codec.build_codec(codec_pth, "cuda")
audio, sr = read_wav(wav)
audio = torchaudio.functional.resample(audio.mean(0, keepdim=True), sr, 44100).cuda()
with torch.no_grad():
    codes, _ = codec.encode(audio)
    print("codes", tuple(codes.shape), codes[0, :, :6].tolist())
    ref = codec.from_indices(codes.clone())[0, 0].float().cpu().numpy()
codes_np = codes[0].T.contiguous().cpu().numpy().astype("<i4")
codes_np.tofile(os.path.join(work, "codes.bin"))
np.save(os.path.join(work, "ref_audio.npy"), ref)
write_wav(os.path.join(work, "ref.wav"), ref)
print("reference decode:", ref.shape)
if codec_st:
    out = os.path.join(work, "rust.wav")
    subprocess.run(["target/release/drama", "decode", "--codec", codec_st, "--codes", os.path.join(work, "codes.bin"), "--out", out], check=True)
    rust = read_wav(out)[0][0].numpy()
    n = min(len(rust), len(ref))
    err = rust[:n] - ref[:n]
    snr = 10 * np.log10((ref[:n] ** 2).sum() / (err ** 2).sum())
    print(f"rust {len(rust)} samples vs ref {len(ref)}; SNR {snr:.1f} dB; max abs err {np.abs(err).max():.4f}")
