"""One decode -> many encoders -> site-level embeddings (DESIGN.md 1.3 "embed", 4.6 steps 1-3).

Each site is fetched and decoded once (raw uint16 and illumination-corrected float32, native
resolution); every encoder then applies its own preprocessing adapter, which resamples to the
common pixel size (R7, ``SiteContext.scale``), and its crop grid; the site embedding is the mean
of the tile CLS embeddings. Output per encoder: ``<out>/<encoder>/sites.parquet`` with metadata
columns and ``e0..eD`` as float16.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.gallery import Gallery
from ..data.shards import ShardReader, load_index
from ..data.sites import IllumCache, correct_site, read_raw_site
from .encoders import Encoder, SiteContext, ViTEncoder

META = [
    "source",
    "batch",
    "plate",
    "well",
    "site",
    "row",
    "col",
    "compound_id",
    "is_dmso",
    "is_anchor",
    "eval_dmso",
    "plate_split",
    "eval_set",
]


def _meta(r) -> dict:
    return {c: getattr(r, c) for c in META if hasattr(r, c)}


def site_embeddings_frame(meta: list[dict], embs: list[np.ndarray]) -> pd.DataFrame:
    X = np.stack(embs).astype(np.float16)
    df = pd.DataFrame(meta)
    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(X, columns=[f"e{i}" for i in range(X.shape[1])])], axis=1
    )


@torch.no_grad()
def _embed_tiles(enc: Encoder, tile_lists: list[np.ndarray], batch_size: int) -> list[np.ndarray]:
    """Embed the tiles of several sites in batches; return one mean vector per site."""
    counts = [len(t) for t in tile_lists]
    allt = np.concatenate(tile_lists)
    outs = []
    for i in range(0, len(allt), batch_size):
        outs.append(enc.embed(torch.from_numpy(np.ascontiguousarray(allt[i : i + batch_size]))).cpu().numpy())
    E = np.concatenate(outs)
    res, k = [], 0
    for c in counts:
        res.append(E[k : k + c].mean(0))
        k += c
    return res


def embed_manifest(
    rows: pd.DataFrame,
    encoders: list[Encoder],
    g: Gallery,
    plate_norms: dict,
    out_dir: Path,
    batch_size: int = 64,
    workers: int = 16,
    sites_per_chunk: int = 32,
    *,
    resample_info: dict,
) -> dict:
    """``resample_info``: ``resample.resample_record(cfg.data, sources)``; its ``resample_scales``
    must cover every source in ``rows`` (it sets each site's ``SiteContext.scale``)."""
    out_dir = Path(out_dir)
    scales = resample_info["resample_scales"]
    missing = set(rows.source) - set(scales)
    if missing:
        raise KeyError(f"no resample scale for sources {sorted(missing)}")
    il = IllumCache(g)
    metas: list[dict] = []
    embs: dict[str, list[np.ndarray]] = {e.name: [] for e in encoders}
    t0, t_dec, t_enc = time.time(), 0.0, {e.name: 0.0 for e in encoders}

    def load(r):
        raw = read_raw_site(g, r)
        return r, raw, correct_site(raw, il, r)

    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(rows), sites_per_chunk):
            ta = time.time()
            chunk = list(ex.map(load, list(rows.iloc[i : i + sites_per_chunk].itertuples())))
            t_dec += time.time() - ta
            ctxs = [
                SiteContext(raw, cor, plate_norms.get(r.plate), _meta(r), scales[r.source])
                for r, raw, cor in chunk
            ]
            for enc in encoders:
                tb = time.time()
                tiles = [enc.preprocess(c) for c in ctxs]
                embs[enc.name] += _embed_tiles(enc, tiles, batch_size)
                t_enc[enc.name] += time.time() - tb
            metas += [c.meta for c in ctxs]
    info = {
        "sites": len(metas),
        "seconds": time.time() - t0,
        "decode_seconds": t_dec,
        "encode_seconds": t_enc,
        "bytes_downloaded": g.bytes_downloaded,
        **resample_info,
    }
    for enc in encoders:
        d = out_dir / enc.name
        d.mkdir(parents=True, exist_ok=True)
        site_embeddings_frame(metas, embs[enc.name]).to_parquet(d / "sites.parquet", index=False)
        (d / "embed_info.json").write_text(
            json.dumps(
                {**info, "encoder": enc.name, "crop": enc.crop, "grid": enc.grid, "window": enc.window},
                indent=1,
            ),
            encoding="utf-8",
        )
    return info


def embed_dev_shards(
    enc: ViTEncoder, dev_dir: str | Path, batch_size: int = 64, sites_per_chunk: int = 16
) -> pd.DataFrame:
    """Embed the dev cache (already plate-normalised uint8 896^2 sites) with one ViT encoder."""
    idx = load_index(dev_dir)
    reader = ShardReader(dev_dir)
    metas, embs = [], []
    for i in range(0, len(idx), sites_per_chunk):
        part = idx.iloc[i : i + sites_per_chunk]
        tiles = [
            enc.preprocess_u8(reader.read(r.shard, int(r.offset), int(r.size))) for r in part.itertuples()
        ]
        embs += _embed_tiles(enc, tiles, batch_size)
        metas += part.drop(columns=["key", "shard", "offset", "size", "shape"]).to_dict("records")
    return site_embeddings_frame(metas, embs)
