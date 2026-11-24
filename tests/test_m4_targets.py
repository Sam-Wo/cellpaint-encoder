"""M4 target annotation (DESIGN.md 4.3, E3): target table mapping rules, multilabel label_retrieval
with its label-shuffle null, pipeline plumbing, Holm family membership, config hash and the
metadata fetcher (local file:// URLs only, no network)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellpaint_encoder.config import (
    HASH_IF_NOT_DEFAULT,
    NON_SCIENCE_KEYS,
    _flatten,
    config_hash,
    load_config,
    to_dict,
)
from cellpaint_encoder.eval import retrieval
from cellpaint_encoder.eval.pipeline import evaluate_wells
from cellpaint_encoder.eval.retrieval import label_retrieval, unit_rows
from cellpaint_encoder.eval.targets import annotation_counts, build_target_table, load_target2, parse_targets
from synthetic_wells import make_wells

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fixture_table(ids=None):
    t2 = load_target2(FIX / "target2_mini.tsv")
    comp = pd.read_csv(FIX / "compound_mini.csv")
    if ids is None:
        ids = list(comp.Metadata_JCP2022)
    return build_target_table(t2, comp, ids)


# ---------------------------------------------------------------- mapping rules


def test_parse_targets():
    assert parse_targets("A|B| A") == {"A", "B"}
    assert parse_targets(np.nan) == parse_targets(None) == parse_targets("") == frozenset()
    assert parse_targets({"x"}) == {"x"}


def test_mapping_rules_full_key_fallback_controls_union():
    table, rep = _fixture_table()
    t = table.set_index("compound_id")
    # full-key join; quinine/quinidine-style duplicate InChIKey -> union of target lists
    assert t.loc["JCP2022_000002", "targets"] == "T1|T3" and t.loc["JCP2022_000002", "join_rule"] == "full"
    # stereo block differs: connectivity-block (14 char) fallback, unique match only
    assert t.loc["JCP2022_000005", "join_rule"] == "connectivity14"
    assert t.loc["JCP2022_000005", "inchikey"] == "EEEEEEEEEEEEEE-ABCDEFGHIJ-N"
    assert "JCP2022_000007" not in t.index  # 14-char block matches two Target-2 keys: ambiguous
    assert "JCP2022_000009" not in t.index  # not in Target-2
    assert "JCP2022_000008" not in t.index  # positive control excluded
    assert "JCP2022_033924" not in t.index  # DMSO excluded
    assert "JCP2022_000006" not in t.index  # treatment without targets
    assert set(t.index) == {f"JCP2022_00000{i}" for i in (1, 2, 3, 4, 5)}
    assert rep["mapped_full"] == 6 and rep["mapped_connectivity14"] == 1
    assert rep["unmapped"] == 2 and rep["unmapped_detail"] == {"unmapped": 1, "unmapped_ambiguous14": 1}
    assert rep["excluded_controls"] == 1 and rep["treatment_without_targets"] == 1
    assert rep["e3_compounds_excl_dmso"] == 9 and rep["e3_dmso_present"]
    assert rep["target2_control_rows"] == 2 and rep["target2_treatment_inchikeys"] == 8
    assert rep["join_rule"] == "full, then connectivity14 fallback"
    # T1 held by A, B, C (3); T2 by A, E; T3 by B, C; T9 by D -> one target, three queries
    assert rep["n_targets_ge3"] == 1 and rep["n_queries_ge3"] == 3 and rep["annotated_compounds"] == 5


def test_mapping_full_only_when_all_match():
    _, rep = _fixture_table(["JCP2022_000001", "JCP2022_000002", "JCP2022_033924"])
    assert (
        rep["join_rule"] == "full" and rep["mapped_connectivity14"] == 0 and rep["annotated_treatments"] == 2
    )


def test_annotation_counts_threshold():
    c = annotation_counts(["a|b", "a", "a|c", "b|c", np.nan, ""], min_members=3)
    assert c == {"annotated_compounds": 4, "n_targets_any": 3, "n_targets_ge3": 1, "n_queries_ge3": 3}


# ---------------------------------------------------------------- multilabel retrieval


def _planted(n_cpd=60, n_targets=12, dim=32, signal=3.0, seed=0, null=False):
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(n_targets, dim))
    labels, X = [], []
    for _ in range(n_cpd):
        k = rng.integers(1, 3)
        ts = rng.choice(n_targets, size=k, replace=False)
        labels.append("|".join(f"G{t}" for t in ts))
        X.append((0 if null else signal) * D[ts].sum(0) + rng.normal(size=dim))
    return np.array(labels, dtype=object), np.stack(X)


def test_multilabel_planted_signal_beats_shuffle_null():
    lab, X = _planted()
    _, s, q = label_retrieval(lab, X, multilabel=True, n_null=200, n_shuffle=100)
    assert s["mAP"] > 0.7
    assert s["mAP"] > s["shuffle_null_mean"] + 10 * s["shuffle_null_sd"]
    assert s["p_shuffle"] == pytest.approx(1 / 101)
    assert s["n_queries"] == len(q) and s["n_targets"] == s["n_classes"]
    assert s["mAP"] == pytest.approx(q.ap.mean())  # headline = mean AP over queries


def test_multilabel_null_data_near_shuffle_null():
    lab, X = _planted(null=True, seed=3)
    _, s, q = label_retrieval(lab, X, multilabel=True, n_null=200, n_shuffle=100)
    assert abs(s["mAP"] - s["shuffle_null_mean"]) < 3 * s["shuffle_null_sd"] + 0.02
    assert s["p_shuffle"] > 0.01
    # the shuffle null is close to the random-ranking null as well
    rnd = np.mean(
        [retrieval.null_ap(r.n_pos, r.n_total, 300, np.random.default_rng(0)).mean() for r in q.itertuples()]
    )
    assert abs(s["shuffle_null_mean"] - rnd) < 0.05


def test_multilabel_small_targets_excluded_but_compounds_stay_candidates():
    rng = np.random.default_rng(1)
    #           0      1      2      3          4      5      6 (unannotated)  7 (empty)
    lab = np.array(["A", "A|B", "A", "B|C", "C", "D", np.nan, ""], dtype=object)
    X = rng.normal(size=(8, 8))
    per, s, q = label_retrieval(lab, X, multilabel=True, n_null=50)
    # counted targets: A (3 compounds); B, C (2 each) and D (1) are below 3
    assert s["n_targets"] == 1 and s["n_annotated"] == 6
    assert sorted(q["query"]) == [0, 1, 2]  # 3, 4, 5 hold no counted target: not queries
    assert (q.n_total == 5).all()  # ... but remain candidates (6 annotated - the query itself)
    assert (q.n_pos == 2).all()  # 1 and 3 share only B (not counted) -> not positives of each other
    assert list(per["query"]) == [0, 1, 2] and set(per.label) == {"A"}


def test_multilabel_positive_on_any_shared_counted_target():
    rng = np.random.default_rng(2)
    lab = np.array(["A|B", "A", "A", "B", "B", "C"], dtype=object)
    _, s, q = label_retrieval(lab, rng.normal(size=(6, 4)), multilabel=True, n_null=20)
    qq = q.set_index("query")
    assert qq.loc[0, "n_pos"] == 4  # shares A with 1, 2 and B with 3, 4
    assert qq.loc[3, "n_pos"] == 2  # shares B with 0, 4
    assert 5 not in qq.index


def test_shuffle_identity_reproduces_observed_and_m3_unchanged():
    lab, X = _planted(seed=5)
    idx, _, _, L = retrieval._membership(list(lab), 3)
    U = unit_rows(X[idx].astype(np.float64))
    _, s, _ = label_retrieval(lab, X, multilabel=True, n_null=20)
    assert retrieval._headline(U @ U.T, L, per_class=False) == pytest.approx(s["mAP"])
    # single label (M3): shuffle off by default, headline unchanged when it is on
    rng = np.random.default_rng(0)
    single = np.repeat(["a", "b", "c", "d"], 5).astype(object)
    Xs = rng.normal(size=(4, 16))[np.repeat(np.arange(4), 5)] * 4 + rng.normal(size=(20, 16))
    _, s0, _ = label_retrieval(single, Xs, n_null=50)
    _, s1, _ = label_retrieval(single, Xs, n_null=50, n_shuffle=30)
    assert "shuffle_null_mean" not in s0 and s1["mAP"] == s0["mAP"] and s1["shuffle_null_mean"] < s0["mAP"]
    Ls = np.stack([single == c for c in "abcd"], axis=1)
    Us = unit_rows(Xs)
    assert retrieval._headline(Us @ Us.T, Ls, per_class=True) == pytest.approx(s0["mAP"])


def test_shuffle_null_permutations_depend_only_on_seed(monkeypatch):
    """Same seed -> same permutations for any representation (X), so methods share one null draw."""
    lab, X = _planted(seed=6)
    seen = []
    orig = np.random.default_rng

    def spy(*a, **k):
        g = orig(*a, **k)
        if a and isinstance(a[0], list):  # the shuffle stream
            seen.append(g.permutation(1000)[:5].tolist())
            g = orig(*a, **k)
        return g

    monkeypatch.setattr(retrieval.np.random, "default_rng", spy)
    _, a, _ = label_retrieval(lab, X, multilabel=True, n_null=20, n_shuffle=20, seed=4)
    Y = np.random.RandomState(0).normal(size=X.shape)  # another representation
    _, b, _ = label_retrieval(lab, Y, multilabel=True, n_null=20, n_shuffle=20, seed=4)
    assert len(seen) == 2 and seen[0] == seen[1]
    _, c, _ = label_retrieval(lab, X, multilabel=True, n_null=20, n_shuffle=20, seed=4)
    assert c["shuffle_null_mean"] == a["shuffle_null_mean"]  # deterministic


# ---------------------------------------------------------------- pipeline + evaluate.py


def _cfg(extra=()):
    return load_config(None, ["eval.n_null=100", "eval.n_boot=200", "eval.n_label_shuffle=30", *extra])


def _with_e3(signal=4.0, seed=11):
    meta, X = make_wells(sources=("source_2", "source_3", "source_8"), cpd_signal=4.0, n_dmso=10, seed=8)
    t2, XT = make_wells(
        sources=("source_2", "source_3", "source_10"), n_cpd=24, cpd_signal=0.0, n_dmso=10, seed=9
    )
    t2["eval_set"], t2["is_anchor"] = "E3", False
    t2["compound_id"] = t2.compound_id.str.replace("CPD", "T2_")
    t2["plate"] = t2.plate + "_T2"
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(6, XT.shape[1]))
    tg = {f"T2_{i}": [i % 6, (i + 1) % 6] if i % 4 == 0 else [i % 6] for i in range(22)}  # 22, 23: controls
    for r in np.flatnonzero(~t2.is_dmso.to_numpy()):
        c = t2.compound_id.iloc[r]
        if c in tg:
            XT[r] = XT[r] + signal * D[tg[c]].sum(0)
    targets = pd.DataFrame(
        {"compound_id": list(tg), "targets": ["|".join(f"G{t}" for t in v) for v in tg.values()]}
    )
    allm = pd.concat([meta, t2], ignore_index=True)
    return allm, np.concatenate([X, XT]), targets


def test_pipeline_m4_target_annotation_planted_and_controls():
    allm, allX, targets = _with_e3()
    cfg = _cfg(["eval.m1_sources=[source_2, source_3, source_8]"])
    res = evaluate_wells(allm, allX, cfg, targets=targets, which=("m4",))
    s = res["M4_target_annotation"]
    assert s["mAP"] > s["shuffle_null_mean"] + 0.2 and s["p_shuffle"] < 0.05
    assert s["n_annotated"] == 22 and s["n_e3_compounds"] == 24  # T2_22, T2_23 unannotated (controls)
    assert s["n_targets"] == 6 and s["sources"] == ["source_10", "source_2", "source_3"]
    q = res["_tables"]["M4_target_annotation_queries"]
    assert not set(q.compound_id) & {"T2_22", "T2_23"} and len(q) == s["n_queries"] == 22
    assert set(res["_tables"]["M4_target_annotation"].compound_id) == set(q.compound_id)
    # same E3 compounds without signal: near the shuffle null
    nul = evaluate_wells(*_with_e3(signal=0.0)[:2], cfg, targets=targets, which=("m4",))
    n = nul["M4_target_annotation"]
    assert abs(n["mAP"] - n["shuffle_null_mean"]) < 3 * n["shuffle_null_sd"] + 0.03


def test_pipeline_m4_target_annotation_skipped_without_table():
    allm, allX, _ = _with_e3()
    res = evaluate_wells(allm, allX, _cfg(), which=("m4",))
    assert res["M4_target_annotation"] == {"skipped": "eval.target_table not set"}
    assert "M4_target2" in res


def test_holm_family_membership():
    ev = _script("evaluate")
    comps = {
        k: {"p_wilcoxon": p}
        for k, p in (
            ("P1_vs_cellprofiler", 0.001),
            ("P2_vs_imagenet", 0.01),
            ("P3_M1_noninferiority_vs_cellprofiler", 0.02),
            ("M3_moa_vs_cellprofiler", 0.03),
            ("M4_target2_vs_cellprofiler", 0.04),
            ("M4_target_annotation_vs_cellprofiler", 0.05),
            ("M5_cross_dose_vs_cellprofiler", 0.06),
        )
    }
    comps["P4_position_inflation"] = {"ours": 0.1, "cellprofiler": 0.2}
    fam = ev.secondary_family(comps)
    assert set(fam) == {
        "P2_vs_imagenet",
        "P3_M1_noninferiority_vs_cellprofiler",
        "M3_moa_vs_cellprofiler",
        "M4_target2_vs_cellprofiler",
        "M4_target_annotation_vs_cellprofiler",
        "M5_cross_dose_vs_cellprofiler",
    }
    assert "P1_vs_cellprofiler" not in fam and "P4_position_inflation" not in fam
    del comps["M4_target_annotation_vs_cellprofiler"]  # absent member (no target table): family of 5
    assert len(ev.secondary_family(comps)) == 5


def test_evaluate_load_targets(tmp_path, capsys):
    ev = _script("evaluate")
    assert ev.load_targets(_cfg()) is None and "skipped" in capsys.readouterr().out
    p = tmp_path / "t.csv"
    pd.DataFrame({"compound_id": ["a"], "targets": ["X|Y"]}).to_csv(p, index=False)
    assert list(ev.load_targets(_cfg([f"eval.target_table={p.as_posix()}"])).targets) == ["X|Y"]
    with pytest.raises(FileNotFoundError):
        ev.load_targets(_cfg([f"eval.target_table={(tmp_path / 'nope.csv').as_posix()}"]))


def test_new_eval_keys_enter_config_hash_only_when_set():
    """Configs that leave the M4b keys at their defaults keep their pre-M4b hash; configs that set
    them (full.yaml and everything based on it, since the prereg-v1 freeze) hash them."""
    for path in [None, *sorted((ROOT / "configs").rglob("*.yaml"))]:
        cfg = load_config(path)
        items = _flatten(to_dict(cfg))
        flat = {k: v for k, v in items.items() if k not in NON_SCIENCE_KEYS and k not in HASH_IF_NOT_DEFAULT}
        legacy = hashlib.sha256(json.dumps(flat, sort_keys=True).encode()).hexdigest()[:10]
        at_default = all(items[k] == d for k, d in HASH_IF_NOT_DEFAULT.items())
        assert (config_hash(cfg) == legacy) == at_default, path
    assert config_hash(load_config(None, ["eval.target_table=x.csv"])) != config_hash(load_config(None))


# ---------------------------------------------------------------- fetcher (file:// only)


def test_fetch_idempotent_manifest_and_loud_failure(tmp_path):
    fj = _script("fetch_jump_metadata")
    src = tmp_path / "remote" / "org" / "repo" / "abc"
    src.mkdir(parents=True)
    (src / "a.tsv").write_bytes(b"x\ty\n1\t2\n")
    files = {"a.tsv": ("org/repo", "abc", "MIT", "a.tsv")}
    base = (tmp_path / "remote").as_uri()
    out = tmp_path / "jump"
    r = fj.fetch_all(out, base=base, files=files)
    entry = json.loads((out / "MANIFEST.json").read_text())["files"]["a.tsv"]
    assert entry["bytes"] == 8 and entry["sha256"] == hashlib.sha256(b"x\ty\n1\t2\n").hexdigest()
    assert r["bytes_downloaded"] == 8
    assert fj.fetch_all(out, base=base, files=files)["bytes_downloaded"] == 0  # unchanged: no download
    (out / "a.tsv").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="does not match"):
        fj.fetch_all(out, base=base, files=files)
    assert fj.fetch_all(out, base=base, files=files, force=True)["bytes_downloaded"] == 8
    with pytest.raises(RuntimeError, match="cannot fetch|HTTP"):
        fj.fetch_all(tmp_path / "j2", base=base, files={"b.tsv": ("org/repo", "abc", "MIT", "missing.tsv")})
    assert not (tmp_path / "j2" / "b.tsv").exists() and not (tmp_path / "j2" / "b.tsv.part").exists()


def test_e3_compound_ids_from_plate_and_well_tables():
    fj = _script("fetch_jump_metadata")
    plates = pd.DataFrame(
        {
            "Metadata_Source": ["source_2", "source_2", "source_3", "source_4"],
            "Metadata_Plate": ["P1", "P2", "P3", "P4"],
            "Metadata_PlateType": ["TARGET2", "COMPOUND", "TARGET2", "TARGET2"],
        }
    )
    wells = pd.DataFrame(
        {
            "Metadata_Source": ["source_2"] * 3 + ["source_3"] * 2 + ["source_4"],
            "Metadata_Plate": ["P1", "P1", "P2", "P3", "P3", "P4"],
            "Metadata_JCP2022": ["J1", "JCP2022_033924", "J9", "J1", "J2", "J7"],
        }
    )
    ids, info = fj.e3_compound_ids(wells, plates, ["source_2", "source_3"])
    assert ids == ["J1", "J2"] and info["missing_by_source"] == {"source_2": ["J2"]}
