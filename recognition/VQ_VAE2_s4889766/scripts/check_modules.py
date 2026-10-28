"""
Sanity checks for modules.py: latent shapes, straight-through gradients and
codebook behaviour. Runs on random input, no data needed.

Usage: python scripts/check_modules.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import VQVAE2  # noqa: E402

torch.manual_seed(0)
x = torch.rand(2, 1, 256, 128)

# Shapes
for levels, expected in [(2, {"top": (2, 32, 16), "bottom": (2, 64, 32)}),
                         (1, {"bottom": (2, 64, 32)})]:
    model = VQVAE2(levels=levels)
    out = model(x)
    assert out["recon"].shape == x.shape, out["recon"].shape
    shapes = {k: tuple(v.shape) for k, v in out["codes"].items()}
    assert shapes == expected, shapes
    n_params = sum(p.numel() for p in model.parameters())
    print(f"levels={levels}: codes {shapes}, recon {tuple(out['recon'].shape)}, "
          f"{n_params / 1e6:.2f}M params")

model = VQVAE2(levels=2)
assert model.reconstruct_top_only(x).shape == x.shape


def grad_norms(model, loss):
    model.zero_grad()
    loss.backward()
    return {name: (m.net[0].weight.grad.abs().sum().item()
                   if m.net[0].weight.grad is not None else 0.0)
            for name, m in [("enc_b", model.enc_b), ("enc_t", model.enc_t),
                            ("dec", model.dec)]}


# Reconstruction loss alone can only reach the encoders through the
# straight-through estimator, since argmin and the codebook lookup have no gradient.
model.train()
g = grad_norms(model, F.mse_loss(model(x)["recon"], x))
assert g["enc_b"] > 0 and g["enc_t"] > 0, g
print(f"recon loss grads: enc_b {g['enc_b']:.3g}, enc_t {g['enc_t']:.3g}")

# Commitment loss trains the encoders only, never the decoder.
out = model(x)
g = grad_norms(model, sum(out["commitment"].values()))
assert g["enc_b"] > 0 and g["enc_t"] > 0 and g["dec"] == 0, g
print("commitment loss grads reach both encoders and not the decoder")

# The codebook is a buffer updated by EMA, never by the optimiser.
assert not any("embed" in n for n, _ in model.named_parameters())

# EMA updates and dead-code restarts happen in train mode only.
q = model.quant_b
before = q.embed.clone()
model.eval()
model(x)
assert torch.equal(before, q.embed), "codebook changed in eval mode"
model.train()
model(x)
assert not torch.equal(before, q.embed), "codebook did not update in train mode"
assert (q.cluster_size >= q.dead_threshold).all(), "dead codes were not restarted"
print("codebook: frozen in eval, EMA-updated in train, dead codes restarted")

# Interpolation endpoints are the plain reconstructions of the two inputs.
for levels in (2, 1):
    model = VQVAE2(levels=levels).eval()
    frames, codes = model.interpolate(x[:1], x[1:], torch.linspace(0, 1, 5))
    recon = model(x)["recon"]
    assert frames.shape == (5, 1, 256, 128)
    assert torch.allclose(frames[0], recon[0], atol=1e-5)
    assert torch.allclose(frames[-1], recon[1], atol=1e-5)
    assert set(codes) == set(model.level_names) and len(codes["bottom"]) == 5
print("interpolation: endpoints match the reconstructions, at both levels settings")

print("All module checks passed.")
