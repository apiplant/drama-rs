"""Build the Fish codec exactly as sglang-omni's modded_dac_vq.yaml does, without hydra."""
import functools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import modded_dac as md  # noqa: E402
import rvq  # noqa: E402


def build_codec(path=None, device="cuda"):
    general = functools.partial(
        md.ModelArgs, block_size=8192, n_local_heads=-1, head_dim=64, rope_base=10000, norm_eps=1e-5,
        dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True,
    )
    tcfg = lambda: md.ModelArgs(block_size=2048, n_layer=8, n_head=16, dim=1024, intermediate_size=3072, n_local_heads=-1,
                                head_dim=64, rope_base=10000, norm_eps=1e-5, dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True)
    post = md.WindowLimitedTransformer(causal=True, window_size=128, input_dim=1024, config=tcfg())
    pre = md.WindowLimitedTransformer(causal=True, window_size=128, input_dim=1024, config=tcfg())
    quantizer = rvq.DownsampleResidualVectorQuantize(
        input_dim=1024, n_codebooks=9, codebook_size=1024, codebook_dim=8, quantizer_dropout=0.5,
        downsample_factor=[2, 2], post_module=post, pre_module=pre, semantic_codebook_size=4096,
    )
    codec = md.DAC(
        sample_rate=44100, encoder_dim=64, encoder_rates=[2, 4, 8, 8], decoder_dim=1536, decoder_rates=[8, 8, 4, 2],
        encoder_transformer_layers=[0, 0, 0, 4], decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=general, quantizer=quantizer,
    )
    if path:
        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        missing, unexpected = codec.load_state_dict(sd, strict=False, assign=True)
        print("missing:", len(missing), missing[:5], "unexpected:", len(unexpected), unexpected[:5])
    return codec.eval().to(device)
