"""
VQ-VAE-2 (Razavi et al., 2019) for single-channel 256 x 128 MRI slices.

The layout follows the paper and rosinality's reference implementation. The
bottom encoder downsamples 4x (64 x 32 codes) and the top encoder a further 2x
(32 x 16 codes). The bottom quantiser is conditioned on the decoded top
latents, and the decoder sees both levels. With levels=1 only the bottom level
is built, giving a single-level VQ-VAE baseline.
"""
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizerOutput(NamedTuple):
    quantized: torch.Tensor   # (B, D, H, W), straight-through gradients to the input
    codes: torch.Tensor       # (B, H, W) codebook indices
    commitment: torch.Tensor  # beta * ||z_e - sg[e]||^2
    perplexity: torch.Tensor  # exp(entropy) of code usage in this batch


class VectorQuantizerEMA(nn.Module):
    """
    Vector quantiser whose codebook is updated by exponential moving averages
    (VQ-VAE paper, Appendix A.1) rather than by a codebook loss.

    codebook_size: number of codes K
    code_dim: dimension D of each code
    beta: commitment loss weight
    decay: EMA decay for the codebook statistics
    eps: Laplace smoothing for the cluster sizes
    dead_code_restart: replace codes whose EMA usage falls below dead_threshold
        (roughly "used less than once per batch") with random encoder outputs
        from the current batch
    """

    def __init__(self, codebook_size: int, code_dim: int, beta: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5, dead_code_restart: bool = True,
                 dead_threshold: float = 1.0):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.dead_code_restart = dead_code_restart
        self.dead_threshold = dead_threshold

        embed = torch.randn(codebook_size, code_dim)
        self.register_buffer("embed", embed)
        # Starting cluster sizes at 1 (not 0) keeps embed == embed_avg / cluster_size
        # consistent and stops every code looking dead while the EMA warms up.
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        b, d, h, w = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)

        distances = (flat.pow(2).sum(1, keepdim=True)
                     - 2 * flat @ self.embed.t()
                     + self.embed.pow(2).sum(1))
        codes = distances.argmin(1)
        counts = torch.bincount(codes, minlength=self.codebook_size).to(flat.dtype)

        if self.training:
            self._ema_update(flat.detach(), codes, counts)

        quantized = F.embedding(codes, self.embed).view(b, h, w, d).permute(0, 3, 1, 2)
        commitment = self.beta * F.mse_loss(z, quantized.detach())
        quantized = z + (quantized - z).detach() # straight-through estimator

        probs = counts / counts.sum()
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        return QuantizerOutput(quantized, codes.view(b, h, w), commitment, perplexity)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, codes: torch.Tensor, counts: torch.Tensor):
        embed_sum = torch.zeros_like(self.embed).index_add_(0, codes, flat)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        self.embed.copy_(self.embed_avg / smoothed.unsqueeze(1))

        if self.dead_code_restart:
            # torch.where rather than boolean indexing avoids a GPU sync every step
            dead = (self.cluster_size < self.dead_threshold).unsqueeze(1)
            samples = flat[torch.randint(len(flat), (self.codebook_size,), device=flat.device)]
            self.embed.copy_(torch.where(dead, samples, self.embed))
            self.embed_avg.copy_(torch.where(dead, samples, self.embed_avg))
            self.cluster_size.copy_(torch.where(dead.squeeze(1),
                                                torch.ones_like(self.cluster_size),
                                                self.cluster_size))

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Look up (B, H, W) codes as (B, D, H, W) quantised vectors."""
        return F.embedding(codes, self.embed).permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    """ReLU, 3x3 conv, ReLU, 1x1 conv, plus the identity."""

    def __init__(self, channels: int, residual_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, residual_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(residual_channels, channels, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """Strided convolutions (stride 2 or 4 overall) followed by residual blocks."""

    def __init__(self, in_channels: int, channels: int, n_res_blocks: int,
                 residual_channels: int, stride: int):
        super().__init__()
        if stride == 4:
            layers = [
                nn.Conv2d(in_channels, channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels // 2, channels, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            ]
        elif stride == 2:
            layers = [
                nn.Conv2d(in_channels, channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels // 2, channels, 3, padding=1),
            ]
        else:
            raise ValueError(f"stride must be 2 or 4, got {stride}")
        layers += [ResBlock(channels, residual_channels) for _ in range(n_res_blocks)]
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Residual blocks followed by transposed convolutions (stride 2 or 4 overall)."""

    def __init__(self, in_channels: int, out_channels: int, channels: int,
                 n_res_blocks: int, residual_channels: int, stride: int):
        super().__init__()
        layers = [nn.Conv2d(in_channels, channels, 3, padding=1)]
        layers += [ResBlock(channels, residual_channels) for _ in range(n_res_blocks)]
        layers.append(nn.ReLU())
        if stride == 4:
            layers += [
                nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(channels // 2, out_channels, 4, stride=2, padding=1),
            ]
        elif stride == 2:
            layers.append(nn.ConvTranspose2d(channels, out_channels, 4, stride=2, padding=1))
        else:
            raise ValueError(f"stride must be 2 or 4, got {stride}")
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class VQVAE2(nn.Module):
    """
    Hierarchical VQ-VAE with one (bottom) or two (top and bottom) latent levels.

    forward() returns a dict with the reconstruction and, per level name
    ("top", "bottom"), the codes, commitment loss and batch perplexity.
    """

    def __init__(self, in_channels: int = 1, hidden_channels: int = 128,
                 residual_channels: int = 64, n_res_blocks: int = 2,
                 codebook_size: int = 512, code_dim: int = 64, beta: float = 0.25,
                 ema_decay: float = 0.99, dead_code_restart: bool = True, levels: int = 2):
        super().__init__()
        if levels not in (1, 2):
            raise ValueError(f"levels must be 1 or 2, got {levels}")
        self.levels = levels

        def quantizer():
            return VectorQuantizerEMA(codebook_size, code_dim, beta, ema_decay,
                                      dead_code_restart=dead_code_restart)

        self.enc_b = Encoder(in_channels, hidden_channels, n_res_blocks,
                             residual_channels, stride=4)
        if levels == 2:
            self.enc_t = Encoder(hidden_channels, hidden_channels, n_res_blocks,
                                 residual_channels, stride=2)
            self.pre_quant_t = nn.Conv2d(hidden_channels, code_dim, 1)
            self.quant_t = quantizer()
            self.dec_t = Decoder(code_dim, code_dim, hidden_channels, n_res_blocks,
                                 residual_channels, stride=2)
            self.upsample_t = nn.ConvTranspose2d(code_dim, code_dim, 4, stride=2, padding=1)

        self.pre_quant_b = nn.Conv2d(hidden_channels + (code_dim if levels == 2 else 0),
                                     code_dim, 1)
        self.quant_b = quantizer()
        self.dec = Decoder(code_dim * levels, in_channels, hidden_channels, n_res_blocks,
                           residual_channels, stride=4)

    @property
    def level_names(self) -> tuple[str, ...]:
        return ("top", "bottom") if self.levels == 2 else ("bottom",)

    def encode(self, x: torch.Tensor) -> dict[str, QuantizerOutput]:
        h_b = self.enc_b(x)
        out = {}
        if self.levels == 2:
            out["top"] = self.quant_t(self.pre_quant_t(self.enc_t(h_b)))
            h_b = torch.cat([self.dec_t(out["top"].quantized), h_b], dim=1)
        out["bottom"] = self.quant_b(self.pre_quant_b(h_b))
        return out

    def decode(self, quant_b: torch.Tensor, quant_t: torch.Tensor | None = None) -> torch.Tensor:
        if self.levels == 2:
            quant_b = torch.cat([self.upsample_t(quant_t), quant_b], dim=1)
        return self.dec(quant_b)

    def decode_codes(self, codes_b: torch.Tensor,
                     codes_t: torch.Tensor | None = None) -> torch.Tensor:
        """Decode from code indices, e.g. those sampled from a prior."""
        quant_t = self.quant_t.embed_codes(codes_t) if self.levels == 2 else None
        return self.decode(self.quant_b.embed_codes(codes_b), quant_t)

    def forward(self, x: torch.Tensor) -> dict:
        enc = self.encode(x)
        quant_t = enc["top"].quantized if self.levels == 2 else None
        return {
            "recon": self.decode(enc["bottom"].quantized, quant_t),
            "codes": {k: v.codes for k, v in enc.items()},
            "commitment": {k: v.commitment for k, v in enc.items()},
            "perplexity": {k: v.perplexity for k, v in enc.items()},
        }

    @torch.no_grad()
    def reconstruct_top_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode with the bottom latents zeroed, to show what the top level alone
        captures (as in the paper's per-level reconstructions).
        """
        if self.levels != 2:
            raise ValueError("Top-only reconstruction needs levels=2")
        enc = self.encode(x)
        return self.decode(torch.zeros_like(enc["bottom"].quantized), enc["top"].quantized)
