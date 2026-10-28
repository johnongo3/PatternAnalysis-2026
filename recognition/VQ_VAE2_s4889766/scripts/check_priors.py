"""
Sanity checks for the priors in modules.py: output shapes, causality (no
position sees itself or anything after it), no blind spot, conditioning, the
row-cropped sampling shortcut, and seeded sampling. Runs on random codes.

Usage: python scripts/check_priors.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules import BottomPrior, TopPrior, build_prior  # noqa: E402

torch.manual_seed(0)
K = 512
small = dict(channels=32, layers=4, kernel_size=5, heads=4, dropout=0.3)
top = TopPrior(K, (32, 16), attention_every=2, **small).eval()
bottom = BottomPrior(K, (64, 32), cond_res_blocks=2, **small).eval()

top_codes = torch.randint(K, (2, 32, 16))
bottom_codes = torch.randint(K, (2, 64, 32))
assert top(top_codes).shape == (2, K, 32, 16)
assert bottom(bottom_codes, top_codes).shape == (2, K, 64, 32)
print("shapes: top logits (B, 512, 32, 16), bottom logits (B, 512, 64, 32)")


def logits_at(model, codes, pos, *cond):
    return model(codes, *cond)[:, :, pos[0], pos[1]]


def check_causal(name, model, codes, *cond):
    rows, cols = codes.shape[1:]
    for pos in [(0, 0), (0, 5), (7, 0), (rows // 2, cols // 2), (rows - 1, cols - 1)]:
        base = logits_at(model, codes, pos, *cond)
        # randomise the current position and everything after it in raster order
        future = torch.zeros(rows, cols, dtype=torch.bool)
        future.view(-1)[pos[0] * cols + pos[1]:] = True
        changed = torch.where(future, torch.randint(K, codes.shape), codes)
        assert torch.allclose(base, logits_at(model, changed, pos, *cond), atol=1e-5), \
            f"{name}: position {pos} sees itself or a later code"

        # no blind spot: the code just above and to the right must matter
        i, j = pos
        if i > 0 and j + 1 < cols:
            moved = codes.clone()
            moved[:, i - 1, j + 1] = (moved[:, i - 1, j + 1] + 1) % K
            assert not torch.allclose(base, logits_at(model, moved, pos, *cond)), \
                f"{name}: blind spot at {pos}"
    print(f"{name}: causal at every tested position, no blind spot above-right")


check_causal("top prior", top, top_codes)
check_causal("bottom prior", bottom, bottom_codes, top_codes)

single_cfg = {"data": {"size": [256, 128]}, "vqvae": {"levels": 1, "codebook_size": K},
              "prior_single": dict(small, attention_every=2)}
single = build_prior("bottom", single_cfg).eval()
assert isinstance(single, TopPrior) and single.shape == (64, 32)
check_causal("single-level prior", single, bottom_codes)

other_top = torch.randint(K, top_codes.shape)
assert not torch.allclose(bottom(bottom_codes, top_codes), bottom(bottom_codes, other_top))
print("bottom prior: output depends on the top codes")

# Sampling only runs the network on rows 0..i; those rows must match the full map.
for name, model, codes, cond in [("top", top, top_codes, None),
                                 ("bottom", bottom, bottom_codes,
                                  bottom.condition(top_codes))]:
    full = model.logits(codes, cond)
    for i in (0, 5, codes.shape[1] - 1):
        part = model.logits(codes[:, :i + 1], cond[:, :, :i + 1] if cond is not None else None)
        assert torch.allclose(full[:, :, :i + 1], part, atol=1e-5), f"{name} crop row {i}"
print("row-cropped logits match the full map")

# Seeded sampling is reproducible and in range. Tiny maps keep this quick.
tiny_top = TopPrior(K, (4, 2), attention_every=2, **small).eval()
tiny_bottom = BottomPrior(K, (8, 4), cond_res_blocks=1, **small).eval()


def draw(seed, temperature):
    g = torch.Generator().manual_seed(seed)
    t = tiny_top.sample(3, temperature, g)
    return t, tiny_bottom.sample(t, temperature, g)


t1, b1 = draw(0, 0.9)
t2, b2 = draw(0, 0.9)
t3, _ = draw(1, 0.9)
assert torch.equal(t1, t2) and torch.equal(b1, b2), "same seed gave different samples"
assert not torch.equal(t1, t3), "different seeds gave the same samples"
assert t1.shape == (3, 4, 2) and b1.shape == (3, 8, 4)
assert 0 <= int(b1.min()) and int(b1.max()) < K
try:
    tiny_top.train().sample(1)
    raise AssertionError("sampling in train mode should fail")
except RuntimeError:
    tiny_top.eval()
print("sampling: reproducible per seed, refuses to run with dropout on")

print("All prior checks passed.")
