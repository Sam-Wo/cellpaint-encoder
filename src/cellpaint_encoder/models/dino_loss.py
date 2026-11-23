"""DINO loss: teacher centering + sharpening, cross-entropy over (teacher global, student other) pairs.

Centering follows DINO (``center = m * center + (1 - m) * batch_mean(teacher_out)``, m = 0.9),
but the update is split from the forward pass: with gradient accumulation the trainer sums the
teacher outputs over all micro-batches and calls :meth:`update_center` **once per optimiser
step** (DESIGN.md 3.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(self, out_dim: int, n_crops: int, student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.n_crops, self.student_temp, self.center_momentum = n_crops, student_temp, center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor, teacher_temp: float):
        s = (student_out.float() / self.student_temp).chunk(self.n_crops)
        t = F.softmax((teacher_out.float() - self.center) / teacher_temp, dim=-1).detach().chunk(2)
        total, n = 0.0, 0
        for iq, q in enumerate(t):
            for v in range(self.n_crops):
                if v == iq:
                    continue
                total = total + torch.sum(-q * F.log_softmax(s[v], dim=-1), dim=-1).mean()
                n += 1
        return total / n

    @torch.no_grad()
    def update_center(self, teacher_sum: torch.Tensor, count: int) -> None:
        batch_center = teacher_sum.float().view(1, -1) / max(count, 1)
        self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))


@torch.no_grad()
def collapse_monitors(student_out, teacher_out, center, teacher_temp, student_temp, feats) -> dict:
    """Teacher entropy (per sample, and of the batch-mean distribution), KL(student||teacher) on
    the global crops, and the per-dimension std of backbone features (DESIGN.md 3.3)."""
    t = F.softmax((teacher_out.float() - center) / teacher_temp, dim=-1)
    n_glob = teacher_out.shape[0]
    s = F.softmax(student_out[:n_glob].float() / student_temp, dim=-1)
    ent = -(t * torch.log(t + 1e-12)).sum(-1).mean()
    tm = t.mean(0)
    ent_mean = -(tm * torch.log(tm + 1e-12)).sum()
    kl = (s * (torch.log(s + 1e-12) - torch.log(t + 1e-12))).sum(-1).mean()
    fstd = feats.float().std(0).mean() if feats.shape[0] > 1 else torch.tensor(float("nan"))
    return {
        "teacher_entropy": float(ent),
        "teacher_mean_entropy": float(ent_mean),
        "kl_student_teacher": float(kl),
        "feat_std": float(fstd),
        "max_entropy": float(torch.log(torch.tensor(float(t.shape[-1])))),
    }
