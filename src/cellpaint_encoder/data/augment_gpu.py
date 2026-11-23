"""Batched multi-crop microscopy augmentation on the training device (DESIGN.md 3.2, 3.5).

Input: uint8 tiles ``[B, 5, 448, 448]``. Output: ``2`` global crops ``[B, 5, 224, 224]`` and
``n_local`` local crops ``[B, 5, 96, 96]``, normalised per channel.

Geometry: each crop is a square window of side ``size * s`` with ``s ~ U(1 - j, 1 + j)``
(``j = scale_jitter = 0.2``, "native resolution with +-20 % scale jitter" replacing DINO's
RandomResizedCrop), placed uniformly inside the tile, combined with a random element of the
dihedral group D4 (flips and 90 degree rotations), and resampled to ``size`` in one
``affine_grid``/``grid_sample`` call.

Intensity (per crop; parameters per channel): with p=0.5, gamma U(0.8, 1.25) then gain
U(0.8, 1.25) and offset N(0, 0.05); Gaussian blur sigma U(0.1, 1.0) with p = 1.0 / 0.1 / 0.5 for
global crop 1 / global crop 2 / local crops; Gaussian noise sigma 0.02 with p = 0.3; channel
dropout (zero one random channel) with p = 0.1. Nothing mixes channels; no colour jitter,
solarisation, large rescaling or shear.

All randomness comes from an explicit ``torch.Generator``; the training loop seeds it from
``(seed, micro_batch_index)`` so augmentation is a pure function of the step (exact resume).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..config import AugCfg

# the 8 elements of D4 as 2x2 matrices acting on (x, y) normalised coordinates
_D4 = torch.tensor(
    [
        [[1, 0], [0, 1]],
        [[0, -1], [1, 0]],
        [[-1, 0], [0, -1]],
        [[0, 1], [-1, 0]],
        [[-1, 0], [0, 1]],
        [[1, 0], [0, -1]],
        [[0, 1], [1, 0]],
        [[0, -1], [-1, 0]],
    ],
    dtype=torch.float32,
)


def _u(g: torch.Generator, shape, lo, hi, device):
    return torch.rand(shape, generator=g, device=device) * (hi - lo) + lo


def random_crops(x: torch.Tensor, size: int, jitter: float, g: torch.Generator) -> torch.Tensor:
    """x: float ``[B, C, H, W]`` -> ``[B, C, size, size]`` (random window, scale jitter, D4)."""
    B, C, H, W = x.shape
    dev = x.device
    s = _u(g, (B,), 1 - jitter, 1 + jitter, dev)
    side = torch.clamp(size * s, max=min(H, W))  # source-pixel side of the window
    half_x, half_y = side / W, side / H  # half-extent in normalised coords ([-1, 1] spans W)
    cx = (_u(g, (B,), 0, 1, dev) * 2 - 1) * (1 - half_x)
    cy = (_u(g, (B,), 0, 1, dev) * 2 - 1) * (1 - half_y)
    d4 = _D4.to(dev)[torch.randint(0, 8, (B,), generator=g, device=dev)]
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, :2] = d4[:, 0] * half_x[:, None]
    theta[:, 1, :2] = d4[:, 1] * half_y[:, None]
    theta[:, 0, 2], theta[:, 1, 2] = cx, cy
    grid = F.affine_grid(theta, (B, C, size, size), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def gaussian_blur(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Per-sample separable Gaussian blur; ``sigma[b] <= 0`` leaves sample b unchanged."""
    B, C, H, W = x.shape
    r = max(1, int(math.ceil(3 * float(sigma.max().clamp(min=0.1)))))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    sg = sigma.clamp(min=1e-3).to(x.dtype)[:, None]
    k = torch.exp(-0.5 * (t[None] / sg) ** 2)
    k = k / k.sum(1, keepdim=True)
    ident = torch.zeros_like(k)
    ident[:, r] = 1
    k = torch.where((sigma > 0)[:, None], k, ident)
    k = k.repeat_interleave(C, 0)  # [B*C, K]
    y = x.reshape(1, B * C, H, W)
    y = F.pad(y, (r, r, 0, 0), mode="reflect")
    y = F.conv2d(y, k[:, None, None, :], groups=B * C)
    y = F.pad(y, (0, 0, r, r), mode="reflect")
    y = F.conv2d(y, k[:, None, :, None], groups=B * C)
    return y.reshape(B, C, H, W)


def intensity_aug(x: torch.Tensor, cfg: AugCfg, p_blur: float, g: torch.Generator) -> torch.Tensor:
    B, C = x.shape[:2]
    dev = x.device
    x = x.clamp(0, 1)
    on = (torch.rand(B, generator=g, device=dev) < cfg.p_intensity)[:, None, None, None]
    gamma = _u(g, (B, C, 1, 1), *cfg.gamma, dev)
    gain = _u(g, (B, C, 1, 1), *cfg.gain, dev)
    offset = torch.randn((B, C, 1, 1), generator=g, device=dev) * cfg.offset_std
    x = torch.where(on, x.clamp(min=1e-6) ** gamma * gain + offset, x)
    sig = _u(g, (B,), *cfg.blur_sigma, dev)
    sig = torch.where(torch.rand(B, generator=g, device=dev) < p_blur, sig, torch.zeros_like(sig))
    if bool((sig > 0).any()):
        x = gaussian_blur(x, sig)
    noise_on = (torch.rand(B, generator=g, device=dev) < cfg.p_noise)[:, None, None, None]
    x = x + noise_on * torch.randn(x.shape, generator=g, device=dev) * cfg.noise_std
    drop_on = torch.rand(B, generator=g, device=dev) < cfg.p_channel_dropout
    ch = torch.randint(0, C, (B,), generator=g, device=dev)
    mask = torch.ones(B, C, 1, 1, device=dev)
    mask[drop_on, ch[drop_on]] = 0
    return x * mask


class MultiCropAugment:
    def __init__(
        self,
        cfg_aug: AugCfg,
        global_size: int,
        local_size: int,
        n_local: int,
        jitter: float,
        mean: list[float],
        std: list[float],
    ):
        self.cfg, self.gs, self.ls, self.nl, self.j = cfg_aug, global_size, local_size, n_local, jitter
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    @torch.no_grad()
    def __call__(self, tiles_u8: torch.Tensor, g: torch.Generator) -> list[torch.Tensor]:
        x = tiles_u8.float() / 255.0
        crops = []
        p_blur = [self.cfg.p_blur_global1, self.cfg.p_blur_global2]
        for i in range(2):
            c = random_crops(x, self.gs, self.j, g)
            crops.append(self.normalise(intensity_aug(c, self.cfg, p_blur[i], g)))
        for _ in range(self.nl):
            c = random_crops(x, self.ls, self.j, g)
            crops.append(self.normalise(intensity_aug(c, self.cfg, self.cfg.p_blur_local, g)))
        return crops
