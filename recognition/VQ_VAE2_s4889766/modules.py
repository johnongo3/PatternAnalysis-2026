"""
VQ-VAE-2 (Razavi et al., 2019) for single-channel 256 x 128 MRI slices.

Stage 1, VQVAE2: the layout follows the paper and rosinality's reference
implementation. The bottom encoder downsamples 4x (64 x 32 codes) and the top
encoder a further 2x (32 x 16 codes). The bottom quantiser is conditioned on the
decoded top latents, and the decoder sees both levels. With levels=1 only the
bottom level is built, giving a single-level VQ-VAE baseline.

Stage 2, TopPrior and BottomPrior: autoregressive priors over the code maps,
built from gated PixelCNN layers (van den Oord et al., 2016) with vertical and
horizontal stacks, so there is no blind spot. The top prior adds causal
self-attention for global context. The bottom prior is conditioned on the top
codes through a residual conditioning stack.
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


def _gate(x: torch.Tensor) -> torch.Tensor:
    """Gated activation tanh(a) * sigmoid(b), splitting channels in half."""
    a, b = x.chunk(2, dim=1)
    return torch.tanh(a) * torch.sigmoid(b)


def _shift_down(x: torch.Tensor) -> torch.Tensor:
    """Shift a feature map down one row, so row i only carries rows above it."""
    return F.pad(x, (0, 0, 1, 0))[:, :, :-1]


class MaskedConv2d(nn.Conv2d):
    """
    Convolution whose kernel is multiplied by a fixed causal mask.

    mask="vertical": k x k kernel seeing the current row and the rows above
    mask="A": 1 x k kernel seeing only positions to the left (first layer)
    mask="B": 1 x k kernel seeing positions to the left and the current one
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, mask: str):
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        k = kernel_size
        if mask == "vertical":
            super().__init__(in_channels, out_channels, (k, k), padding=k // 2)
            m = torch.zeros(k, k)
            m[: k // 2 + 1] = 1
        elif mask in ("A", "B"):
            super().__init__(in_channels, out_channels, (1, k), padding=(0, k // 2))
            m = torch.zeros(1, k)
            m[:, : k // 2 + (mask == "B")] = 1
        else:
            raise ValueError(f"mask must be 'vertical', 'A' or 'B', got {mask!r}")
        self.register_buffer("mask", m)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding)


class GatedResBlock(nn.Module):
    """
    One gated PixelCNN layer. The vertical stack sees every row above, and is
    shifted down a row before feeding the horizontal stack, which sees the
    current row to the left. Together they cover all earlier positions in
    raster order. Optional conditioning (B, cond_channels, H, W) is projected
    and added to both stacks.

    mask_type: "A" for the first layer (excludes the current code), else "B"
    residual: add the horizontal input back; must be False for the first layer,
        whose input still contains the current code
    """

    def __init__(self, channels: int, kernel_size: int, mask_type: str, residual: bool,
                 dropout: float = 0.0, cond_channels: int = 0):
        super().__init__()
        self.residual = residual
        self.v_conv = MaskedConv2d(channels, 2 * channels, kernel_size, "vertical")
        self.h_conv = MaskedConv2d(channels, 2 * channels, kernel_size, mask_type)
        self.v_to_h = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.h_out = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.cond = nn.Conv2d(cond_channels, 2 * channels, 1) if cond_channels else None

    def forward(self, v, h, cond=None):
        v_pre = self.v_conv(v)
        h_pre = self.h_conv(h) + self.v_to_h(_shift_down(v_pre))
        if self.cond is not None:
            c = self.cond(cond)
            v_pre, h_pre = v_pre + c, h_pre + c
        h_new = self.h_out(self.dropout(_gate(h_pre)))
        return _gate(v_pre), (h + h_new if self.residual else h_new)


class CausalAttention(nn.Module):
    """
    Multi-head self-attention over a feature map in raster order, where each
    position attends to itself and earlier positions only. Applied to the
    horizontal stack, whose features at a position already exclude that code.

    Normalisation is LayerNorm over channels at each position; normalising
    across positions (BatchNorm, GroupNorm) would leak future codes.
    """

    def __init__(self, channels: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels ({channels}) must be divisible by heads ({heads})")
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2) # (B, H*W, C), raster order
        q, k, v = (self.qkv(self.norm(seq))
                   .view(b, h * w, 3, self.heads, c // self.heads)
                   .permute(2, 0, 3, 1, 4))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.dropout if self.training else 0.0)
        out = self.proj(out.transpose(1, 2).reshape(b, h * w, c))
        return x + out.transpose(1, 2).view(b, c, h, w)


class ConditioningStack(nn.Module):
    """
    Turns top codes into a feature map on the bottom grid: embedding, residual
    blocks at 32 x 16, then a 2x transposed convolution to 64 x 32. It need not
    be causal, since the whole top map is known before the bottom is generated.
    """

    def __init__(self, codebook_size: int, channels: int, n_res_blocks: int):
        super().__init__()
        self.embed = nn.Embedding(codebook_size, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            *[ResBlock(channels, channels) for _ in range(n_res_blocks)],
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
        )

    def forward(self, top_codes: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(top_codes).permute(0, 3, 1, 2))


class PixelCNNPrior(nn.Module):
    """
    Gated PixelCNN over a (H, W) map of codebook indices, predicting each code
    from the ones before it in raster order. Shared by TopPrior and BottomPrior.

    codebook_size: number of codes K (the output classes)
    shape: (H, W) of the code map
    attention_every: add a CausalAttention block after every this many gated
        layers; 0 disables attention
    cond_channels: channels of the conditioning map passed to every layer, or 0
    """

    def __init__(self, codebook_size: int, shape: tuple[int, int], channels: int = 256,
                 layers: int = 12, kernel_size: int = 5, attention_every: int = 0,
                 heads: int = 4, dropout: float = 0.0, cond_channels: int = 0):
        super().__init__()
        self.codebook_size = codebook_size
        self.shape = tuple(shape)
        self.embed = nn.Embedding(codebook_size, channels)
        self.pos = nn.Parameter(torch.randn(1, channels, *shape) * 0.02)
        self.blocks = nn.ModuleList(
            GatedResBlock(channels, kernel_size, "A" if i == 0 else "B", residual=i > 0,
                          dropout=dropout, cond_channels=cond_channels)
            for i in range(layers))
        self.attention = nn.ModuleDict({
            str(i): CausalAttention(channels, heads, dropout)
            for i in range(layers) if attention_every and (i + 1) % attention_every == 0})
        self.head = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 1),
                                  nn.ReLU(), nn.Conv2d(channels, codebook_size, 1))

    def logits(self, codes: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        (B, H', W) codes to (B, K, H', W) logits. H' may be fewer rows than the
        full map: every layer is causal, so the first rows do not depend on later ones.
        """
        x = self.embed(codes).permute(0, 3, 1, 2) + self.pos[:, :, :codes.shape[1]]
        v = h = x
        for i, block in enumerate(self.blocks):
            v, h = block(v, h, cond)
            if str(i) in self.attention:
                h = self.attention[str(i)](h)
        return self.head(h)

    @torch.no_grad()
    def _sample(self, n: int, temperature: float, generator: torch.Generator | None,
                cond: torch.Tensor | None) -> torch.Tensor:
        """Sample n code maps one position at a time, in raster order."""
        if self.training:
            raise RuntimeError("Call .eval() before sampling, so dropout is off")
        rows, cols = self.shape
        codes = torch.zeros(n, rows, cols, dtype=torch.long, device=self.pos.device)
        for i in range(rows):
            # rows below i cannot affect row i, so only run the network on rows 0..i
            row_cond = cond[:, :, :i + 1] if cond is not None else None
            for j in range(cols):
                logits = self.logits(codes[:, :i + 1], row_cond)[:, :, i, j] / temperature
                codes[:, i, j] = torch.multinomial(F.softmax(logits, dim=1), 1,
                                                   generator=generator).squeeze(1)
        return codes


class TopPrior(PixelCNNPrior):
    """
    Unconditional prior with causal attention: over the top codes of the
    two-level model, or over the only (bottom) codes of the single-level baseline.
    """

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.logits(codes)

    def sample(self, n: int, temperature: float = 1.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
        return self._sample(n, temperature, generator, None)


class BottomPrior(PixelCNNPrior):
    """
    Prior over the bottom code map, conditioned on the top codes. Trained with
    teacher forcing on the true top codes; at sampling time the top codes come
    from TopPrior.
    """

    def __init__(self, codebook_size: int, shape: tuple[int, int], channels: int = 256,
                 layers: int = 10, kernel_size: int = 5, attention_every: int = 0,
                 heads: int = 4, dropout: float = 0.0, cond_res_blocks: int = 8):
        super().__init__(codebook_size, shape, channels, layers, kernel_size,
                         attention_every, heads, dropout, cond_channels=channels)
        self.condition = ConditioningStack(codebook_size, channels, cond_res_blocks)

    def forward(self, codes: torch.Tensor, top_codes: torch.Tensor) -> torch.Tensor:
        return self.logits(codes, self.condition(top_codes))

    def sample(self, top_codes: torch.Tensor, temperature: float = 1.0,
               generator: torch.Generator | None = None) -> torch.Tensor:
        return self._sample(len(top_codes), temperature, generator, self.condition(top_codes))


def build_vqvae(cfg: dict) -> VQVAE2:
    """Build a VQVAE2 from the vqvae section of a config."""
    kwargs = {k: v for k, v in cfg["vqvae"].items() if k != "ssim_loss_weight"}
    return VQVAE2(in_channels=1, **kwargs)


def prior_section(level: str, cfg: dict) -> str:
    """
    The config section holding a prior's architecture: prior_top and
    prior_bottom for a two-level model, prior_single for the single-level baseline.
    """
    return "prior_single" if cfg["vqvae"]["levels"] == 1 else f"prior_{level}"


def build_prior(level: str, cfg: dict) -> PixelCNNPrior:
    """
    Build a prior from a config. A two-level model has a top prior and a bottom
    prior conditioned on the top codes. The single-level baseline has one
    unconditional prior over its bottom codes. The code map shape follows from
    the image size: /8 for top, /4 for bottom.
    """
    rows, cols = cfg["data"]["size"]
    k = cfg["vqvae"]["codebook_size"]
    if cfg["vqvae"]["levels"] == 1:
        if level != "bottom":
            raise ValueError("A single-level model only has bottom codes")
        return TopPrior(k, (rows // 4, cols // 4), **cfg["prior_single"])
    if level == "top":
        return TopPrior(k, (rows // 8, cols // 8), **cfg["prior_top"])
    if level == "bottom":
        return BottomPrior(k, (rows // 4, cols // 4), **cfg["prior_bottom"])
    raise ValueError(f"level must be 'top' or 'bottom', got {level!r}")
