"""Baseline adapters against their documented interfaces (stubs: no NC weights are ever loaded
here), the CellProfiler selection, the source_6 lock and the end-to-end evaluation pipeline."""

import io

import numpy as np
import pandas as pd
import torch

from cellpaint_encoder.config import load_config
from cellpaint_encoder.embed.baselines import deepcellmorpho, gallery_dl, openphenom
from cellpaint_encoder.embed.encoders import SiteContext
from cellpaint_encoder.eval.pipeline import evaluate_wells
from cellpaint_encoder.eval.profiles import BLOCKLIST, select_cp_features
from cellpaint_encoder.models.vit import VisionTransformer
from synthetic_wells import make_wells


def test_deepcellmorpho_adapter_loads_dinov2_layout(tmp_path):
    ref = VisionTransformer(
        img_size=32, patch_size=16, in_chans=5, embed_dim=384, depth=12, num_heads=6, layerscale=True
    )
    sd = {f"teacher.backbone.{k}": v for k, v in ref.state_dict().items()}
    sd["teacher.backbone.mask_token"] = torch.zeros(1, 384)
    sd["teacher.dino_head.mlp.0.weight"] = torch.zeros(4, 4)
    p = tmp_path / "converted.pt"
    torch.save(sd, p)  # what scripts/convert_untrusted_checkpoint.py produces (plain tensors)
    enc = deepcellmorpho.build(str(p), "dinov2_vits16", crop=32, grid=2)
    site = np.random.default_rng(0).integers(0, 4000, (5, 70, 70)).astype(np.uint16)
    tiles = enc.preprocess(SiteContext(site, site.astype(np.float32), None))
    assert tiles.shape == (4, 5, 32, 32) and tiles.min() >= 0 and tiles.max() <= 1
    out = enc.embed(torch.from_numpy(tiles))
    assert out.shape == (4, 384)
    with torch.no_grad():
        ref.eval()
        assert torch.allclose(out, ref(torch.from_numpy(tiles)), atol=1e-5)


def test_openphenom_adapter_with_stub_model():
    class Stub:
        def predict(self, x):
            assert x.dtype == torch.uint8 and x.shape[1:] == (5, 256, 256)
            return x.float().mean((2, 3)).repeat(1, 77)[:, :384]

    enc = openphenom.OpenPhenomEncoder(Stub())
    site = np.random.default_rng(0).integers(0, 4000, (5, 800, 800)).astype(np.uint16)
    tiles = enc.preprocess(SiteContext(site, None, None))
    assert tiles.shape == (9, 5, 256, 256) and tiles.dtype == np.uint8
    assert enc.embed(torch.from_numpy(tiles)).shape == (9, 384)


def test_gallery_dl_profile_parse():
    df = pd.DataFrame(
        {
            "Metadata_Source": ["source_2"] * 2,
            "Metadata_Plate": ["p"] * 2,
            "Metadata_Well": ["A01", "A02"],
            "all_emb": [
                np.arange(672, dtype=np.float32),
                np.random.default_rng(0).random(672).astype(np.float32),
            ],
        }
    )
    b = io.BytesIO()
    df.to_parquet(b)
    out = gallery_dl.parse_profile_parquet(b.getvalue())
    assert out.shape == (2, 3 + 672) and out.e671.iloc[0] == 671
    F = out[[f"e{i}" for i in range(672)]].to_numpy()
    assert gallery_dl.fidelity_r(F, F) > 0.999


def test_cellprofiler_selection_drops_blocklist_and_correlated():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 5))
    X = np.column_stack([base, base[:, 0] * 2 + 1e-3 * rng.normal(size=200), rng.normal(size=200)])
    feats = [f"Cells_F{i}" for i in range(5)] + ["Cells_F0_copy", BLOCKLIST[0]]
    sel, info = select_cp_features(X, np.abs(X).max(0), feats)
    assert BLOCKLIST[0] not in sel and info["drop_blocklist"] == 1
    assert ("Cells_F0" in sel) != ("Cells_F0_copy" in sel) and info["drop_correlated"] == 1


def _cfg(extra=()):
    return load_config(
        None,
        [
            "eval.dev_query_sources=[source_2, source_3, source_8]",
            "eval.dev_reference_sources=[source_2, source_3, source_8]",
            "eval.reference_sources=[source_2, source_3, source_8]",
            "eval.query_source=source_6",
            "eval.n_null=100",
            "eval.n_boot=200",
            *extra,
        ],
    )


def test_pipeline_end_to_end_and_source6_lock():
    meta, X = make_wells(
        sources=("source_2", "source_3", "source_8", "source_6"),
        cpd_signal=4.0,
        n_dmso=10,
        plates_per_source=4,
        seed=7,
    )
    res = evaluate_wells(
        meta,
        X,
        _cfg(),
        moa=pd.DataFrame({"compound_id": [f"CPD{i}" for i in range(12)], "moa": list("aaabbbcccddd")}),
    )
    assert res["locked"] and "source_6" not in set(res["_meta"].source)
    assert "primary" not in res  # locked: only the dev form exists
    assert res["dev_cross_source"]["mAP"] > 0.8
    assert res["eps"] in (0.1, 1.0, 3.0, 10.0)
    for k in ("M1", "M2", "M3_moa", "position_probe", "source_probe_raw", "shapeshift_strict"):
        assert k in res
    un = evaluate_wells(meta, X, _cfg(["eval.unlock_heldout=true"]))
    assert "source_6" in set(un["_meta"].source) and un["primary"]["mAP"] > 0.8
    assert un["primary_strict"]["n_groups"] <= un["primary"]["n_groups"]
