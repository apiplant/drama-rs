# drama-rs

Fish Audio [Drama 3](https://huggingface.co/fishaudio/drama-3) text-to-speech on a single
24 GB GPU (RTX 4090), in Rust + CUDA. No PyTorch at runtime.

- **Backbone**: Qwen3.5-MoE (35B total / 3B active, 40 layers, gated-DeltaNet linear attention
  + full attention every 4th layer). Routed experts are 4-bit, everything else 8-bit.
- **Codebook tower**: 13 transformer layers that predict codebooks 1–9 of every frame.
- **Codec**: the Fish modded-DAC encoder (voice cloning) and decoder, streaming and exact.

## Footprint and speed (RTX 4090)

| | |
|---|---|
| weights on GPU (LM + codec) | ~21.4 GiB of 24 |
| decode | ~5.2 ms / frame ≈ 190 frames/s ≈ **8.9× real time** (21.5 frames = 1 s of audio) |
| prefill | ~140 ms for a 300-row prompt (text + 10 s reference voice) |
| time to first audio (streaming) | ~50–120 ms after the prompt is ready |
| codec decode | ~200× real time |

Each decode step is one CUDA-graph replay: lm_head → sampling (with RAS) → 13 tower layers
with 9 codebook samplers → backbone step for the new frame. Weight-streaming GEMVs prefetch
the next kernel's weights into L2 to hide launch gaps.

## Quantization

`tools/convert.py` reads the original checkpoint shard by shard:

| tensor | format |
|---|---|
| routed experts (32B params, FP8 in the checkpoint) | int4, asymmetric, group 64, f16 scale+min, MSE clip search |
| attention / DeltaNet / shared expert / tower linears | int8, symmetric, group 128 |
| lm_head | only the 4097 sampleable rows (EOS + semantic band), bf16 |
| embeddings, router | bf16 |

## Setup

```bash
# 1. weights (42 GB)
hf download fishaudio/drama-3 --local-dir models/drama-3

# 2. convert (needs a CUDA PyTorch; ~1 min per layer is plenty)
python tools/convert.py models/drama-3 models/drama-3-q4
python tools/convert_codec.py models/drama-3/codec.pth models/drama-3-q4/codec.safetensors
cp models/drama-3/tokenizer.json models/drama-3-q4/

# 3. build (CUDA toolkit in /opt/cuda, or set CUDA_PATH; DRAMA_CUDA_ARCH defaults to sm_89)
cargo build --release
```

## Usage

```bash
# speak to a file
./target/release/drama speak --model models/drama-3-q4 --text "Hello there." --out hello.wav

# stream to the speakers while generating (PipeWire / pw-play)
./target/release/drama speak --model models/drama-3-q4 --text "Hello there." --play

# clone a voice from a reference clip and its transcript
./target/release/drama speak --model models/drama-3-q4 --play \
  --ref-wav voice.wav --ref-text "Exactly what is said in voice.wav." \
  --text "<|speaker:0|> [sighs] Fine. If we miss it, it's on you."

# keep the model loaded and speak every line typed on stdin
./target/release/drama speak --model models/drama-3-q4 --interactive --play
```

Sampling defaults follow the reference server: `--temperature 1.0 --top-k 30 --top-p 0.9`,
plus repetition-aware sampling over the last 16 frames. Multi-speaker text uses
`<|speaker:N|>` tags (N in 0..15); inline tags like `[laughs]` are passed through.

Other commands: `encode` / `decode` (codec only), `bench` (decode speed), `trace` (engine tests).

## Tests

All checks compare against a plain-PyTorch reference (`tools/reference.py`):

```bash
PY=python  # a CUDA PyTorch
$PY tools/make_tiny.py models/drama-3/config.json test/tiny-src --tokenizer models/drama-3/tokenizer.json
$PY tools/convert.py test/tiny-src test/tiny
$PY tools/check_engine.py test/tiny                     # engine logits vs reference, per step
# int4 vs original FP8 (KL, top-1), on a real prompt: random rows make the model emit EOS,
# which never happens mid-utterance in real use and skews the codebook-head comparison
./target/release/drama speak --model models/drama-3-q4 --text "..." --out x.wav --prompt-out test/work/prompt.bin
$PY tools/check_engine.py models/drama-3-q4 --original models/drama-3 --rows-file test/work/prompt.bin --steps 12
$PY tools/check_codec.py models/drama-3/codec.pth clip.wav test/work models/drama-3-q4/codec.safetensors
```

`tools/make_bench_model.py` writes a full-size random model for speed/memory tests without the
download.

## Layout

```
kernels/drama.cu   LM kernels: int8/int4 GEMVs, fused MoE, gated DeltaNet, attention, sampling
kernels/codec.cu   codec kernels: im2col, snake, ConvNeXt, windowed attention, VQ encode
src/engine.rs      prefill (cuBLAS + grouped MoE) and the graph-captured decode step
src/codec.rs       streaming codec decoder, encoder, resampler
src/tts.rs         prompt -> frames -> audio pipeline, LM and codec overlapped
ref/               upstream sources used as the reference (sglang-omni, codec)
```
