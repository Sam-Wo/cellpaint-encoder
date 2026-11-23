"""DINO projection head and the multi-crop wrapper (facebookresearch/dino, Apache-2.0 design).

The weight-normalised last layer is written out explicitly (``last_layer_v`` / ``last_layer_g``)
instead of ``torch.nn.utils.weight_norm`` so the state dict is plain tensors; with
``norm_last_layer=True`` the gain is fixed at 1, as in DINO.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3, norm_last_layer=True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(nlayers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        v = torch.empty(out_dim, bottleneck_dim)
        nn.init.trunc_normal_(v, std=0.02)
        self.last_layer_v = nn.Parameter(v)
        self.last_layer_g = nn.Parameter(torch.ones(out_dim, 1), requires_grad=not norm_last_layer)

    def forward(self, x):
        x = F.normalize(self.mlp(x), dim=-1, p=2)
        w = self.last_layer_g * F.normalize(self.last_layer_v, dim=1)
        return F.linear(x, w)

    def last_layer_params(self):
        return [self.last_layer_v, self.last_layer_g]


class MultiCropWrapper(nn.Module):
    """Runs the backbone once per crop resolution, then the head on all CLS tokens."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, crops: list[torch.Tensor], return_backbone: bool = False):
        sizes = torch.tensor([c.shape[-1] for c in crops])
        ends = torch.cumsum(torch.unique_consecutive(sizes, return_counts=True)[1], 0).tolist()
        feats, start = [], 0
        for end in ends:
            feats.append(self.backbone(torch.cat(crops[start:end])))
            start = end
        f = torch.cat(feats)
        out = self.head(f)
        return (out, f) if return_backbone else out
