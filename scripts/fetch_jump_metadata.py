"""Fetch the pinned JUMP metadata files into data/jump/ and build the E3 target table (M4, DESIGN.md 4.3).

    py -3.13 -I -X utf8 scripts/fetch_jump_metadata.py [--out data/jump] [--config configs/full.yaml]

Downloads (raw.githubusercontent.com, pinned commits, ~14.7 MB in all):

* jump-cellpainting/JUMP-Target @ 40ce99d4: ``JUMP-Target-2_compound_metadata.tsv`` (MIT licence)
* jump-cellpainting/datasets @ 016e865f: ``metadata/{compound,plate,well}.csv.gz`` (BSD-3-Clause)

Each file's SHA-256 and size go to ``<out>/MANIFEST.json``. Idempotent: a file whose bytes match
its MANIFEST entry is not downloaded again; a file that no longer matches its entry is an error
(re-fetch with ``--force``). A file present without an entry is downloaded again unless
``--adopt-existing`` (then it is hashed and recorded as adopted). HTTP errors abort.

Then writes ``<out>/target2_targets.csv`` (``eval.target_table``: compound_id, inchikey, pert_iname,
targets, join_rule) and ``<out>/target2_mapping.json`` (counts), for the E3 compound IDs: every
``Metadata_JCP2022`` on TARGET2 plates (``plate.csv.gz``) of ``data.e3_sources`` (``well.csv.gz``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # works under python -I

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request

import pandas as pd

from cellpaint_encoder.config import load_config
from cellpaint_encoder.eval.targets import DMSO_JCP2022, build_target_table, load_target2

RAW = "https://raw.githubusercontent.com"
JUMP_TARGET = ("jump-cellpainting/JUMP-Target", "40ce99d43f0654947cd990845652654902cd008f", "MIT")
DATASETS = ("jump-cellpainting/datasets", "016e865fa0691244e0860943e41c7d6a88ed2580", "BSD-3-Clause")
FILES = {  # local name -> (repo, commit, licence, path in repo)
    "JUMP-Target-2_compound_metadata.tsv": (*JUMP_TARGET, "JUMP-Target-2_compound_metadata.tsv"),
    "compound.csv.gz": (*DATASETS, "metadata/compound.csv.gz"),
    "plate.csv.gz": (*DATASETS, "metadata/plate.csv.gz"),
    "well.csv.gz": (*DATASETS, "metadata/well.csv.gz"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def url_for(repo: str, commit: str, path: str, base: str = RAW) -> str:
    return f"{base}/{repo}/{commit}/{path}"


def download(url: str, dest: Path, timeout: float = 120) -> int:
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            status = getattr(r, "status", None)
            if status is not None and status != 200:
                raise RuntimeError(f"HTTP {status} for {url}")
            expected = r.headers.get("Content-Length") if hasattr(r, "headers") else None
            n = 0
            with open(tmp, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 20), b""):
                    f.write(chunk)
                    n += len(chunk)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"HTTP {e.code} fetching {url}") from e
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"cannot fetch {url}: {e.reason}") from e
    if expected is not None and int(expected) != n:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url}: got {n} bytes, Content-Length {expected}")
    os.replace(tmp, dest)
    return n


def fetch_all(
    out: Path, force: bool = False, adopt_existing: bool = False, base: str = RAW, files: dict = FILES
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"files": {}}
    entries = manifest.setdefault("files", {})
    downloaded = 0
    for name, (repo, commit, licence, path) in files.items():
        dest = out / name
        url = url_for(repo, commit, path, base)
        entry = entries.get(name)
        if dest.exists() and not force:
            if entry is not None and entry.get("url") == url:
                if sha256(dest) != entry["sha256"] or dest.stat().st_size != entry["bytes"]:
                    raise RuntimeError(f"{dest} does not match MANIFEST.json (re-fetch with --force)")
                print(f"ok      {name} (unchanged)")
                continue
            if adopt_existing:
                entries[name] = {
                    "url": url,
                    "repo": repo,
                    "commit": commit,
                    "path": path,
                    "licence": licence,
                    "sha256": sha256(dest),
                    "bytes": dest.stat().st_size,
                    "adopted_existing": True,
                }
                print(f"adopted {name}")
                continue
        n = download(url, dest)
        downloaded += n
        entries[name] = {
            "url": url,
            "repo": repo,
            "commit": commit,
            "path": path,
            "licence": licence,
            "sha256": sha256(dest),
            "bytes": n,
        }
        print(f"fetched {name} ({n:,} bytes)")
    manifest["licences"] = {JUMP_TARGET[0]: JUMP_TARGET[2], DATASETS[0]: DATASETS[2]}
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return {"bytes_downloaded": downloaded, "manifest": manifest}


def e3_compound_ids(wells: pd.DataFrame, plates: pd.DataFrame, sources) -> tuple[list[str], dict]:
    """JCP2022 IDs on TARGET2 plates of ``sources``, with per-source counts (DMSO excluded)."""
    t2p = plates[(plates.Metadata_PlateType == "TARGET2") & plates.Metadata_Source.isin(list(sources))]
    w = wells[wells.Metadata_Plate.isin(set(t2p.Metadata_Plate))].dropna(subset=["Metadata_JCP2022"])
    w = w[w.Metadata_JCP2022 != DMSO_JCP2022]
    per = w.groupby("Metadata_Source").Metadata_JCP2022.nunique().to_dict()
    allids = set(w.Metadata_JCP2022)
    missing = {
        s: sorted(allids - set(w[w.Metadata_Source == s].Metadata_JCP2022))
        for s in sorted(set(w.Metadata_Source))
    }
    info = {
        "target2_plates_by_source": t2p.groupby("Metadata_Source").size().to_dict(),
        "compounds_by_source": per,
        "missing_by_source": {s: v for s, v in missing.items() if v},
    }
    return sorted(allids), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/jump")
    ap.add_argument("--config", default=None, help="for data.e3_sources (default: DataCfg defaults)")
    ap.add_argument("--force", action="store_true", help="download again even if MANIFEST matches")
    ap.add_argument(
        "--adopt-existing", action="store_true", help="record files already present without an entry"
    )
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    out = Path(a.out)
    try:
        f = fetch_all(out, force=a.force, adopt_existing=a.adopt_existing)
    except RuntimeError as e:
        sys.exit(f"fetch_jump_metadata: {e}")
    t2 = load_target2(out / "JUMP-Target-2_compound_metadata.tsv")
    compounds = pd.read_csv(out / "compound.csv.gz", usecols=["Metadata_JCP2022", "Metadata_InChIKey"])
    wells = pd.read_csv(out / "well.csv.gz")
    plates = pd.read_csv(out / "plate.csv.gz")
    ids, e3info = e3_compound_ids(wells, plates, cfg.data.e3_sources)
    table, report = build_target_table(t2, compounds, ids)
    report = {"e3_sources": list(cfg.data.e3_sources), **e3info, **report}
    report["bytes_downloaded_this_run"] = f["bytes_downloaded"]
    table.to_csv(out / "target2_targets.csv", index=False)
    (out / "target2_mapping.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"wrote {out / 'target2_targets.csv'} ({len(table)} compounds); set eval.target_table to it")


if __name__ == "__main__":
    main()
