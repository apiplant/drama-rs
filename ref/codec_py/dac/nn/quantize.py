# Minimal copy of descript-audio-codec's dac/nn/quantize.py (inference paths)
import torch
import torch.nn.functional as F
from torch import nn

from dac.nn.layers import WNConv1d


class VectorQuantize(nn.Module):
    def __init__(self, input_dim, codebook_size, codebook_dim):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)
        zero = torch.zeros(z.shape[0], device=z.device)
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)
        return z_q, zero, zero, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        b, d, t = latents.shape
        encodings = latents.transpose(1, 2).reshape(b * t, d)
        codebook = self.codebook.weight
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].reshape(b, t)
        return self.decode_code(indices), indices


class ResidualVectorQuantize(nn.Module):
    def __init__(self, input_dim=512, n_codebooks=9, codebook_size=1024, codebook_dim=8, quantizer_dropout=0.0):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, codebook_dim[i]) for i in range(n_codebooks)]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers=None):
        z_q = 0
        residual = z
        codes, latents = [], []
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        for i, quantizer in enumerate(self.quantizers):
            if i >= n_quantizers:
                break
            z_q_i, _, _, indices_i, z_e_i = quantizer(residual)
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            codes.append(indices_i)
            latents.append(z_e_i)
        zero = torch.zeros(z.shape[0], device=z.device)
        return z_q, torch.stack(codes, dim=1), torch.cat(latents, dim=1), zero, zero

    def from_codes(self, codes):
        z_q = 0.0
        z_p = []
        for i in range(codes.shape[1]):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q = z_q + self.quantizers[i].out_proj(z_p_i)
        return z_q, torch.cat(z_p, dim=1), codes
