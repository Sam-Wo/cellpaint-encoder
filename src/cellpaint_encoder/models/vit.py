"""ViT-S/16 with a 5-channel patch embedding, SDPA attention, drop-path (DESIGN.md 3.1).

Parameter names follow facebookresearch/dino ``vision_transformer.py`` exactly, so the ImageNet
checkpoint ``dino_deitsmall16_pretrain.pth`` loads with ``strict=True`` once its patch embedding
has been inflated.

**Channel mapping of the ImageNet initialisation.** The RGB filter bank
``W in R^{384 x 3 x 16 x 16}`` has no correspondence with stains, so no stain is mapped to a
colour. Every stain gets the same filter, the mean over RGB, scaled by 3/5:

    W5[:, c] = mean_rgb(W) * 3 / 5        for c in [DNA, ER, RNA, AGP, Mito]

For an input whose 5 channels all equal a grey image ``g`` the 5-channel response is
``sum_c W5_c * g = 3 * mean_rgb(W) * g``, exactly the RGB response to ``(g, g, g)``: activation
scale is preserved and the first layer starts as a channel-symmetric "luminance" detector that
training then differentiates per stain. ``tests/test_model_shapes.py`` checks this identity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if p == 0.0 or not training:
        return x
    keep = 1 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        return drop_path(x, self.p, self.training)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])  # flash / mem-efficient on GPU
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class LayerScale(nn.Module):  # DINOv2 (``blocks.N.ls1.gamma``); used only by baseline adapters
    def __init__(self, dim, init=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dpr, layerscale: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(dpr)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls1 = LayerScale(dim) if layerscale else nn.Identity()
        self.ls2 = LayerScale(dim) if layerscale else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=5,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        layerscale: bool = False,
    ):
        super().__init__()
        self.patch_size, self.embed_dim = patch_size, embed_dim
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        n = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            Block(embed_dim, num_heads, mlp_ratio, dpr[i], layerscale) for i in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def interpolate_pos_encoding(self, npatch_h: int, npatch_w: int) -> torch.Tensor:
        N = self.pos_embed.shape[1] - 1
        if npatch_h * npatch_w == N and npatch_h == npatch_w:
            return self.pos_embed
        cls, patch = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        side = int(math.sqrt(N))
        patch = patch.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        patch = F.interpolate(patch.float(), size=(npatch_h, npatch_w), mode="bicubic", align_corners=False)
        patch = patch.to(cls.dtype).permute(0, 2, 3, 1).reshape(1, npatch_h * npatch_w, -1)
        return torch.cat([cls, patch], dim=1)

    def prepare_tokens(self, x):
        B, _, H, W = x.shape
        t = self.patch_embed(x)
        t = torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1)
        return t + self.interpolate_pos_encoding(H // self.patch_size, W // self.patch_size)

    def forward_features(self, x):
        t = self.prepare_tokens(x)
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)

    def forward(self, x):
        return self.forward_features(x)[:, 0]  # CLS token


def build_vit(cfg_model, drop_path: float | None = None) -> VisionTransformer:
    if cfg_model.arch not in ("vit_small", "vit_tiny_test", "vit_custom"):
        raise ValueError(f"unknown arch {cfg_model.arch}")
    return VisionTransformer(
        patch_size=cfg_model.patch_size,
        in_chans=cfg_model.in_chans,
        embed_dim=cfg_model.embed_dim,
        depth=cfg_model.depth,
        num_heads=cfg_model.num_heads,
        mlp_ratio=cfg_model.mlp_ratio,
        drop_path_rate=cfg_model.drop_path if drop_path is None else drop_path,
    )


def inflate_patch_embed(w_rgb: torch.Tensor, in_chans: int = 5) -> torch.Tensor:
    """``[D, 3, p, p] -> [D, in_chans, p, p]``: mean over RGB, repeated, scaled by 3/in_chans."""
    if w_rgb.shape[1] != 3:
        raise ValueError(f"expected an RGB patch embedding, got {tuple(w_rgb.shape)}")
    return w_rgb.mean(1, keepdim=True).repeat(1, in_chans, 1, 1) * (3.0 / in_chans)


def load_imagenet_dino(model: VisionTransformer, path: str) -> dict:
    """Load the ImageNet DINO ViT-S/16 backbone (Apache-2.0), inflating the patch embedding.

    ``weights_only=True`` (no arbitrary unpickling); strict key matching after the swap."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "teacher" in sd or "student" in sd:  # full training checkpoint layout
        sd = sd.get("teacher", sd.get("student"))
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items() if "head" not in k}
    sd = dict(sd)
    in_chans = model.patch_embed.proj.weight.shape[1]
    sd["patch_embed.proj.weight"] = inflate_patch_embed(sd["patch_embed.proj.weight"], in_chans)
    res = model.load_state_dict(sd, strict=True)
    return {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys), "n_tensors": len(sd)}
