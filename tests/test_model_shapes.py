import torch

from cellpaint_encoder.config import load_config
from cellpaint_encoder.models.dino_head import DINOHead, MultiCropWrapper
from cellpaint_encoder.models.vit import VisionTransformer, build_vit, inflate_patch_embed, load_imagenet_dino


def tiny(in_chans=5):
    return VisionTransformer(
        img_size=32, patch_size=16, in_chans=in_chans, embed_dim=64, depth=2, num_heads=2
    )


def test_shapes_and_pos_interpolation():
    m = tiny().eval()
    for s in (32, 64, 96, 16):
        assert m(torch.randn(2, 5, s, s)).shape == (2, 64)


def test_vit_small_param_count():
    m = build_vit(load_config(None).model)
    n = sum(p.numel() for p in m.parameters())
    assert 21.5e6 < n < 22.2e6  # ViT-S/16, ~22M (DESIGN 3.1)


def test_inflation_preserves_response():
    """Equal channels: 5-channel response == RGB response of the original filters (DESIGN 3.1)."""
    rgb = tiny(3)
    five = tiny(5)
    sd = rgb.state_dict()
    sd["patch_embed.proj.weight"] = inflate_patch_embed(sd["patch_embed.proj.weight"], 5)
    five.load_state_dict(sd)
    g = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        a = rgb.patch_embed(g.repeat(1, 3, 1, 1))  # original RGB filters on (g, g, g)
        b = five.patch_embed(g.repeat(1, 5, 1, 1))  # inflated filters on 5 x g
        assert torch.allclose(a, b, atol=1e-5)
        # whole-network response is then identical too
        rgb.eval(), five.eval()
        assert torch.allclose(rgb(g.repeat(1, 3, 1, 1)), five(g.repeat(1, 5, 1, 1)), atol=1e-4)


def test_load_imagenet_layout_strict(tmp_path):
    rgb = tiny(3)
    p = tmp_path / "fake_dino.pth"
    torch.save(rgb.state_dict(), p)
    five = tiny(5)
    info = load_imagenet_dino(five, str(p))
    assert info["missing"] == [] and info["unexpected"] == []
    assert five.patch_embed.proj.weight.shape[1] == 5


def test_multicrop_wrapper_and_head():
    head = DINOHead(64, 128, hidden_dim=32, bottleneck_dim=16)
    w = MultiCropWrapper(tiny(), head)
    crops = [torch.randn(3, 5, 32, 32)] * 2 + [torch.randn(3, 5, 16, 16)] * 3
    out, feats = w(crops, return_backbone=True)
    assert out.shape == (15, 128) and feats.shape == (15, 64)
    assert not head.last_layer_g.requires_grad  # norm_last_layer: gain fixed at 1
