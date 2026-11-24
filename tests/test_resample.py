"""Pixel-size resampling (DESIGN.md risk R7): scale factors, geometry, the uint8 cache path, every
pixel-producing path going through ``resample.resample_site``, and the train/eval mismatch guards.
CPU only, no network (a synthetic local mirror stands in for the bucket)."""

import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch

from cellpaint_encoder import CHANNELS
from cellpaint_encoder.config import config_hash, load_config
from cellpaint_encoder.data import resample
from cellpaint_encoder.data.gallery import Gallery, illum_key
from cellpaint_encoder.data.normalize import normalise_site
from cellpaint_encoder.data.sites import IllumCache, read_site, read_site_uint8
from cellpaint_encoder.embed.baselines import deepcellmorpho, openphenom
from cellpaint_encoder.embed.encoders import SiteContext, ViTEncoder
from cellpaint_encoder.embed.run import embed_manifest
from conftest import tiny_overrides

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").rglob("*.yaml"))
ON = ["data.resample_um_per_px=0.65", "data.resample_tol=0.02"]
S3 = 0.597976 / 0.65  # source_3: downsample
S8 = 0.7032 / 0.65  # source_8: upsample


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- configuration
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_resamples_identically(path):
    dc = load_config(path).data
    assert dc.resample_um_per_px == 0.65 and dc.resample_tol == 0.02
    assert dc.pixel_size_um == resample.PIXEL_SIZE_UM and len(dc.pixel_size_um) == 13


def test_scale_factors_per_source():
    dc = load_config(ROOT / "configs/full.yaml").data
    for s in ("source_2", "source_6", "source_7", "source_10", "source_13", "source_5"):
        assert resample.scale_for(dc, s) == 1.0  # within 2%: native pixels
    assert resample.scale_for(dc, "source_3") == pytest.approx(0.919963, abs=1e-6)
    assert resample.scale_for(dc, "source_8") == pytest.approx(1.081846, abs=1e-6)
    assert resample.resampled_shape(1080, 1080, S3) == (994, 994)
    assert resample.resampled_shape(1080, 1280, S8) == (1168, 1385)
    off = load_config(None).data
    assert off.resample_um_per_px is None and resample.scale_for(off, "source_3") == 1.0


def test_missing_source_raises():
    dc = load_config(ROOT / "configs/full.yaml").data
    with pytest.raises(KeyError, match="source_12"):
        resample.scale_for(dc, "source_12")
    with pytest.raises(KeyError):
        resample.resample_record(dc, ["source_2", "source_12"])
    assert resample.scale_for(load_config(None).data, "source_12") == 1.0  # off: no table needed


def test_mapping_key_override_and_hash():
    cfg = load_config(ROOT / "configs/full.yaml", ["data.pixel_size_um.source_3=0.6"])
    assert cfg.data.pixel_size_um["source_3"] == 0.6 and cfg.data.pixel_size_um["source_8"] == 0.7032
    base = load_config(ROOT / "configs/full.yaml")
    assert config_hash(cfg) != config_hash(base)
    assert config_hash(load_config(ROOT / "configs/full.yaml", ["data.resample_tol=0.03"])) != config_hash(
        base
    )
    with pytest.raises(TypeError):
        load_config(None, ["data.pixel_size_um={source_3: abc}"])


def test_pilot_and_full_share_the_run_hash():
    assert config_hash(load_config(ROOT / "configs/pilot.yaml")) == config_hash(
        load_config(ROOT / "configs/full.yaml")
    )


# ------------------------------------------------------------------------------ the function
def test_within_tolerance_is_bitwise_identical():
    rng = np.random.default_rng(0)
    x = rng.random((5, 40, 40), dtype=np.float32) * 300
    assert resample.resample_site(x, 1.0) is x
    lo, hi = np.full(5, 10, np.float32), np.full(5, 250, np.float32)
    norm = {"lo": lo.tolist(), "hi": hi.tolist()}
    dc = load_config(ROOT / "configs/full.yaml").data
    for s in ("source_2", "source_6", "source_10"):
        a = normalise_site(x, "per_plate_dmso", norm, 0.1, 99.9, scale=resample.scale_for(dc, s))
        # the pre-R7 arithmetic, verbatim
        old = (x - lo[:, None, None]) * (255.0 / (hi - lo)[:, None, None])
        np.clip(old, 0, 255, out=old)
        assert np.array_equal(a, np.rint(old).astype(np.uint8))


@pytest.mark.parametrize("scale", [S3, S8])
def test_constant_image_stays_constant(scale):
    x = np.full((5, 101, 87), 37.25, np.float32)
    y = resample.resample_site(x, scale)
    assert y.dtype == np.float32 and y.shape == (5, *resample.resampled_shape(101, 87, scale))
    assert np.allclose(y, 37.25, atol=1e-4)


@pytest.mark.parametrize("native,scale", [(0.597976, S3), (0.7032, S8)])
def test_disc_keeps_its_radius_in_microns(native, scale):
    r_um, n = 15.0, 241
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2
    disc = (((yy - c) ** 2 + (xx - c) ** 2) <= (r_um / native) ** 2).astype(np.float32)
    out = resample.resample_site(disc, scale)
    assert out.ndim == 2 and abs(out.shape[0] - n * scale) <= 0.5
    r_out_px = math.sqrt(float(out.sum()) / math.pi)  # area-equivalent radius in new pixels
    assert abs(r_out_px * 0.65 - r_um) <= 0.65  # within one target pixel
    thr = math.sqrt(float((out > 0.5).sum()) / math.pi)
    assert abs(thr * 0.65 - r_um) <= 0.65
    # the centre stays at the centre (align_corners=False keeps the field geometry)
    cy = (out.sum(1) * np.arange(out.shape[0])).sum() / out.sum()
    assert abs(cy - (out.shape[0] - 1) / 2) < 0.5


# ------------------------------------------------------------------- synthetic mirror + manifest
SIZES = {"source_2": 96, "source_3": 100}


def _tiff(a):
    b = io.BytesIO()
    tifffile.imwrite(b, a)
    return b.getvalue()


def _npy(a):
    b = io.BytesIO()
    np.save(b, a)
    return b.getvalue()


def make_mirror(root: Path, constant: int | None = None) -> pd.DataFrame:
    """Two plates (source_2 native, source_3 resampled): 1 training well, 1 norm/eval DMSO well,
    1 anchor well each, with TIFFs and illumination functions under ``root/<key>``."""
    rng = np.random.default_rng(1)
    rows = []
    for s, n in SIZES.items():
        plate, batch = f"P_{s}", "B1"
        for c in CHANNELS:
            k = root / illum_key(s, batch, plate, c)
            k.parent.mkdir(parents=True, exist_ok=True)
            k.write_bytes(_npy((rng.random((n, n)) * 0.4 + 0.8).astype(np.float32)))
        for j, (well, kind) in enumerate([("A01", "train"), ("A02", "dmso"), ("A03", "anchor")]):
            keys = {}
            for c in CHANNELS:
                key = f"cpg0016-jump/{s}/images/{batch}/images/{plate}/{plate}_{well}_s1_{c}.tiff"
                if constant is None:
                    img = rng.integers(100, 4000, (n, n)).astype(np.uint16)
                    img[n // 4 : n // 2, n // 4 : n // 2] += 20000  # bright foreground block
                else:
                    img = np.full((n, n), constant, np.uint16)
                (root / key).parent.mkdir(parents=True, exist_ok=True)
                (root / key).write_bytes(_tiff(img))
                keys[f"key_{c}"] = key
            rows.append(
                {
                    "source": s,
                    "batch": batch,
                    "plate": plate,
                    "well": well,
                    "site": 1,
                    **keys,
                    "row": 0,
                    "col": j,
                    "compound_id": "C1" if kind == "anchor" else ("DMSO" if kind == "dmso" else "X"),
                    "is_dmso": kind == "dmso",
                    "is_anchor": kind == "anchor",
                    "plate_split": "train",
                    "train": kind == "train",
                    "eval_set": "" if kind == "train" else "E1",
                    "dev": kind != "train",
                    "norm": kind == "dmso",
                    "eval_dmso": kind == "dmso",
                    "dev_dmso": kind == "dmso",
                }
            )
    return pd.DataFrame(rows)


def _data_overrides(tmp_path: Path, shards: str = "train") -> list[str]:
    return [
        f"data.manifest={(tmp_path / 'manifest.parquet').as_posix()}",
        f"data.shards_dir={(tmp_path / 'shards' / shards).as_posix()}",
        f"data.dev_shards_dir={(tmp_path / 'shards' / 'dev').as_posix()}",
        f"data.raw_mirror={(tmp_path / 'raw').as_posix()}",
        "data.byte_cap=1",  # any network GET fails
        "data.train_sources=[source_2, source_3]",
        "data.s3_workers=2",
        "data.tile_size=32",
        "data.tiles_per_site=2",
        "data.fg_min=0.0",
        "data.dev_site_size=64",
    ]


@pytest.fixture
def mirror(tmp_path):
    m = make_mirror(tmp_path / "raw")
    m.to_parquet(tmp_path / "manifest.parquet", index=False)
    return m


class Spy:
    def __init__(self):
        self.calls = []
        self.fn = resample.resample_site

    def __call__(self, x, scale):
        self.calls.append((tuple(x.shape), scale))
        return self.fn(x, scale)

    def scales(self):
        return [s for _, s in self.calls]


def _run_write_shards(tmp_path, overrides, monkeypatch):
    ws = _script("write_shards")
    cfg_path = (ROOT / "configs" / "smoke_cpu.yaml").as_posix()
    monkeypatch.setattr(sys, "argv", ["write_shards.py", "--config", cfg_path, "--procs", "0", *overrides])
    ws.main()


# ------------------------------------------------------------------------ the uint8 cache path
def test_uint8_cache_path_dtype_range_and_agreement(tmp_path):
    m = make_mirror(tmp_path / "raw")
    g = Gallery(mirror=tmp_path / "raw", byte_cap=1)
    il = IllumCache(g)
    row = m[(m.source == "source_3") & m.train].iloc[0]
    lo, hi = np.full(5, 500.0, np.float32), np.full(5, 15000.0, np.float32)
    u8 = read_site_uint8(g, il, row, lo, hi, scale=S3)
    assert u8.dtype == np.uint8 and u8.shape == (5, 92, 92)
    assert u8.min() == 0 and u8.max() == 255  # range preserved, clipped at both ends
    # the fused path and normalise_site(read_site(...)) (per-image ablation / ViT embed) agree
    ref = normalise_site(read_site(g, il, row), "per_plate_dmso", {"lo": lo, "hi": hi}, 0.1, 99.9, scale=S3)
    assert np.abs(u8.astype(int) - ref.astype(int)).max() <= 1
    # constant field in, constant field out (illum = 1 so it stays at one grey level)
    m2 = make_mirror(tmp_path / "raw_const", constant=3000)
    g2 = Gallery(mirror=tmp_path / "raw_const", byte_cap=1)
    r2 = m2[m2.source == "source_3"].iloc[0]
    for c in CHANNELS:  # flat illumination
        (tmp_path / "raw_const" / illum_key(r2.source, r2.batch, r2.plate, c)).write_bytes(
            _npy(np.ones((100, 100), np.float32))
        )
    c8 = read_site_uint8(g2, IllumCache(g2), r2, lo, hi, scale=S3)
    assert c8.shape == (5, 92, 92) and len(np.unique(c8)) == 1


# ------------------------------------------------------------- every path goes through the function
@pytest.mark.parametrize("norm_mode", ["per_plate_dmso", "per_image"])
def test_write_shards_resamples_every_site(tmp_path, mirror, monkeypatch, norm_mode):
    spy = Spy()
    monkeypatch.setattr(resample, "resample_site", spy)
    _run_write_shards(tmp_path, [*_data_overrides(tmp_path), *ON, f"data.norm_mode={norm_mode}"], monkeypatch)
    n3 = int(((mirror.source == "source_3") & (mirror.train | mirror.dev)).sum())
    n2 = int(((mirror.source == "source_2") & (mirror.train | mirror.dev)).sum())
    # per_plate_dmso: fused path, one call per channel; per_image: one call per site (train), and
    # the dev cache always uses the fused per-plate path
    per_site = {"per_plate_dmso": 5, "per_image": None}[norm_mode]
    if per_site:
        assert spy.scales().count(pytest.approx(S3)) == 5 * n3
        assert spy.scales().count(1.0) == 5 * n2
    else:
        n3_train = int(((mirror.source == "source_3") & mirror.train).sum())
        assert spy.scales().count(pytest.approx(S3)) == n3_train + 5 * (n3 - n3_train)
    assert all(s in (1.0, pytest.approx(S3)) for s in spy.scales())
    for kind in ("train", "dev"):
        st = json.loads((tmp_path / "shards" / kind / "stats.json").read_text(encoding="utf-8"))
        assert st["resample_um_per_px"] == 0.65 and st["resample_tol"] == 0.02
        assert st["resample_scales"]["source_2"] == 1.0
        assert st["resample_scales"]["source_3"] == pytest.approx(S3)
    idx = pd.read_parquet(tmp_path / "shards" / "train" / "index.parquet")
    assert set(idx.source) == {"source_2", "source_3"} and all(list(x) == [5, 32, 32] for x in idx["shape"])


def test_write_shards_missing_pixel_size_raises(tmp_path, mirror, monkeypatch):
    ov = [*_data_overrides(tmp_path), *ON, "data.pixel_size_um={source_2: 0.65}"]
    with pytest.raises(KeyError, match="source_3"):
        _run_write_shards(tmp_path, ov, monkeypatch)


def test_train_refuses_mismatched_shards(tmp_path, mirror, monkeypatch):
    from cellpaint_encoder.train.loop import Trainer

    _run_write_shards(tmp_path, [*_data_overrides(tmp_path), *ON], monkeypatch)
    shards = tmp_path / "shards" / "train"
    good = load_config(ROOT / "configs/full.yaml", tiny_overrides(shards, tmp_path / "runs"))
    Trainer(good, tmp_path / "ok", install_signals=False)  # matching: fine
    for bad in (["data.resample_um_per_px=null"], ["data.resample_um_per_px=0.6"], ["data.resample_tol=0.1"]):
        cfg = load_config(ROOT / "configs/full.yaml", tiny_overrides(shards, tmp_path / "runs") + bad)
        with pytest.raises(RuntimeError, match="resample"):
            Trainer(cfg, tmp_path / "bad", install_signals=False)
    # shards built without resampling are refused by a resampling config
    legacy = tmp_path / "legacy"
    from conftest import write_synthetic_shards

    write_synthetic_shards(legacy)
    with pytest.raises(RuntimeError, match="resample_um_per_px=None"):
        Trainer(load_config(ROOT / "configs/full.yaml", tiny_overrides(legacy)), tmp_path / "l", False)
    # scripts/train.py checks the dev cache before training, not at the first dev evaluation
    import signal

    monkeypatch.setattr(signal, "signal", lambda *a: None)
    st_path = tmp_path / "shards" / "dev" / "stats.json"
    st = json.loads(st_path.read_text(encoding="utf-8"))
    st_path.write_text(json.dumps({**st, "resample_tol": 0.05}), encoding="utf-8")  # e.g. `--only dev`
    ov = tiny_overrides(shards, tmp_path / "runs") + [
        f"data.dev_shards_dir={(tmp_path / 'shards' / 'dev').as_posix()}",
        "eval.dev_every_epochs=1",
    ]
    argv = ["train.py", "--config", (ROOT / "configs/full.yaml").as_posix(), "--resume", "none"]
    monkeypatch.setattr(sys, "argv", [*argv, "--run-dir", (tmp_path / "tr").as_posix(), *ov])
    with pytest.raises(RuntimeError, match="dev shards.*resample_tol"):
        _script("train").main()
    assert not (tmp_path / "tr" / "train_log.csv").exists()


class _MeanModel(torch.nn.Module):
    def forward(self, x):
        return x.float().mean((2, 3))


class _PhenomStub:
    def predict(self, x):
        return x.float().mean((2, 3))


def _tiny_vit(name):
    from cellpaint_encoder.models.vit import build_vit

    cfg = load_config(None, ["model.embed_dim=32", "model.depth=1", "model.num_heads=2"])
    return ViTEncoder(name, build_vit(cfg.model, drop_path=0.0), [0.2] * 5, [0.2] * 5, crop=32, grid=2)


def test_embed_every_encoder_resamples(tmp_path, mirror, monkeypatch):
    dc = load_config(ROOT / "configs/full.yaml").data
    g = Gallery(mirror=tmp_path / "raw", byte_cap=1)
    rows = mirror[mirror.eval_set == "E1"].reset_index(drop=True)
    norms = {p: {"lo": [500.0] * 5, "hi": [15000.0] * 5} for p in rows.plate}
    encs = [
        _tiny_vit("ours"),
        _tiny_vit("imagenet_dino"),
        _tiny_vit("random_init"),
        deepcellmorpho.DeepCellMorphoEncoder(_MeanModel(), crop=32, grid=2),
        openphenom.OpenPhenomEncoder(_PhenomStub(), crop=32, grid=2),
    ]
    spy = Spy()
    monkeypatch.setattr(resample, "resample_site", spy)
    rec = resample.resample_record(dc, set(rows.source))
    info = embed_manifest(rows, encs, g, norms, tmp_path / "emb", 8, 2, resample_info=rec)
    n3, n2 = int((rows.source == "source_3").sum()), int((rows.source == "source_2").sum())
    # each of the 5 encoders resamples each site once (whole [5,H,W] float image)
    assert spy.scales().count(pytest.approx(S3)) == 5 * n3 and spy.scales().count(1.0) == 5 * n2
    assert all(shape == (5, 100, 100) for shape, s in spy.calls if s != 1.0)
    assert info["resample_scales"] == {"source_2": 1.0, "source_3": pytest.approx(S3)}
    for e in encs:
        ei = json.loads((tmp_path / "emb" / e.name / "embed_info.json").read_text(encoding="utf-8"))
        assert ei["resample_um_per_px"] == 0.65 and ei["resample_scales"]["source_3"] == pytest.approx(S3)
    with pytest.raises(KeyError):
        embed_manifest(rows, encs[:1], g, norms, tmp_path / "e2", resample_info={"resample_scales": {}})


def test_vit_embed_pixels_equal_dev_cache_pixels(tmp_path, mirror):
    """The ViT adapter and the dev cache (fused path) produce the same resampled site."""
    g = Gallery(mirror=tmp_path / "raw", byte_cap=1)
    il = IllumCache(g)
    row = mirror[(mirror.source == "source_3") & mirror.dev].iloc[0]
    norm = {"lo": [500.0] * 5, "hi": [15000.0] * 5}
    dev = read_site_uint8(g, il, row, norm["lo"], norm["hi"], scale=S3)
    seen = {}

    class Capture(ViTEncoder):
        def __init__(self):
            self.name, self.crop, self.grid, self.window, self.pct = "cap", 46, 2, None, (0.1, 99.9)
            self.norm_mode = "per_plate_dmso"

    import cellpaint_encoder.embed.encoders as encmod

    orig = encmod.grid_tiles
    try:
        encmod.grid_tiles = lambda u8, *a, **k: seen.setdefault("u8", u8)
        Capture().preprocess(SiteContext(None, read_site(g, il, row), norm, {}, S3))
    finally:
        encmod.grid_tiles = orig
    assert seen["u8"].shape == dev.shape and np.abs(seen["u8"].astype(int) - dev.astype(int)).max() <= 1


def test_embed_script_refuses_mismatch(tmp_path, mirror, monkeypatch):
    from cellpaint_encoder.train.loop import Trainer

    base = [*_data_overrides(tmp_path), *ON]
    _run_write_shards(tmp_path, base, monkeypatch)
    small = ["model.embed_dim=32", "model.depth=1", "model.num_heads=2", "eval.crop=32", "eval.grid=2"]
    small += ["run.device=cpu", "run.threads=0"]
    emb = _script("embed")

    def run(extra, enc="random_init", out="emb"):
        cfg_path = (ROOT / "configs" / "smoke_cpu.yaml").as_posix()
        argv = ["embed.py", "--config", cfg_path, "--encoder", enc, "--out", (tmp_path / out).as_posix()]
        monkeypatch.setattr(sys, "argv", [*argv, "--workers", "2", *base, *small, *extra])
        emb.main()

    run([])  # matching settings: works and records the scales
    ei = json.loads((tmp_path / "emb" / "random_init" / "embed_info.json").read_text(encoding="utf-8"))
    assert ei["resample_scales"] == {"source_2": 1.0, "source_3": pytest.approx(S3)}
    for bad in (["data.resample_um_per_px=null"], ["data.resample_um_per_px=0.7"]):
        with pytest.raises(RuntimeError, match="resample"):
            run(bad)
    with pytest.raises(KeyError):
        run(["data.pixel_size_um={source_2: 0.65}"])
    # ours=: the checkpoint's config must match too
    shards = tmp_path / "shards" / "train"
    ov = tiny_overrides(shards, tmp_path / "runs") + ["optim.epochs=1"]
    cfg = load_config(ROOT / "configs/full.yaml", ov + ["data.resample_tol=0.05"])
    with pytest.raises(RuntimeError):  # trainer already refuses these shards
        Trainer(cfg, tmp_path / "t0", install_signals=False)
    tr = Trainer(load_config(ROOT / "configs/full.yaml", ov), tmp_path / "t1", install_signals=False)
    tr.train(resume=None)
    ckpt = (tmp_path / "t1" / "teacher_backbone.pt").as_posix()
    tiny_model = ["model.embed_dim=64", "model.depth=2", "model.num_heads=2"]
    run(tiny_model, enc=f"ours={ckpt}", out="emb_ours")
    with pytest.raises(RuntimeError, match="checkpoint"):
        # current config resamples differently from the run that produced the checkpoint
        run([*tiny_model, "data.resample_tol=0.05", "data.shards_dir=nowhere"], enc=f"ours={ckpt}", out="x")
