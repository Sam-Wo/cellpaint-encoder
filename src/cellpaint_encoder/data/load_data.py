"""Parse JUMP ``load_data(_with_illum).csv`` into one row per site with canonical channels.

Channel order differs by source (Mito is C02 in source_7, C05 in source_10, ch1 in source_3,
ch4 in source_15; DESIGN.md 1.1), so stains are mapped **only** through the ``URL_Orig<Stain>``
column names, never through file names or channel indices. Brightfield columns
(``URL_OrigBrightfield{,_L,_H}``) are dropped: the model uses the five fluorescence stains in
the canonical order ``[DNA, ER, RNA, AGP, Mito]``. A source without one of the five (source_15
has no RNA) raises :class:`MissingChannelError`.
"""

from __future__ import annotations

import io
import re

import pandas as pd

from .. import CHANNELS
from .gallery import key_from_url

META = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_Site"]
SKIP_FILE = re.compile(r"(?:DC_DCAM|_thumb)", re.IGNORECASE)


class MissingChannelError(ValueError):
    pass


def well_to_rc(well: str) -> tuple[int, int]:
    """``'A01'`` -> (0, 0); ``'P24'`` -> (15, 23); 1536-well ``'AA01'`` -> (26, 0)."""
    m = re.fullmatch(r"([A-Z]+)(\d+)", well)
    if not m:
        raise ValueError(f"bad well id {well!r}")
    letters, num = m.groups()
    if len(letters) == 1:
        return ord(letters) - 65, int(num) - 1
    if len(letters) == 2 and letters[0] == "A":  # 'AA' follows 'Z' on 1536-well plates
        return 26 + ord(letters[1]) - 65, int(num) - 1
    raise ValueError(f"bad well id {well!r}")


def parse_load_data(raw: bytes | str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        df = raw
    else:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    missing_meta = [c for c in META if c not in df.columns]
    if missing_meta:
        raise ValueError(f"load_data.csv lacks {missing_meta}")
    missing = [c for c in CHANNELS if f"URL_Orig{c}" not in df.columns]
    if missing:
        src = df["Metadata_Source"].iloc[0] if len(df) else "?"
        raise MissingChannelError(f"{src}: no URL_Orig{missing} column; a 5-channel model cannot use it")
    out = pd.DataFrame({
        "source": df["Metadata_Source"].astype(str),
        "batch": df["Metadata_Batch"].astype(str),
        "plate": df["Metadata_Plate"].astype(str),
        "well": df["Metadata_Well"].astype(str),
        "site": df["Metadata_Site"].astype(int),
    })
    for c in CHANNELS:
        keys = df[f"URL_Orig{c}"].astype(str).map(key_from_url)
        bad = keys[keys.str.contains(SKIP_FILE)]
        if len(bad):
            raise ValueError(f"calibration/thumbnail file mapped as {c}: {bad.iloc[0]}")
        out[f"key_{c}"] = keys
    for c in CHANNELS:
        col = f"URL_Illum{c}"
        if col in df.columns:
            out[f"illum_{c}"] = df[col].astype(str).map(key_from_url)
    rc = out["well"].map(well_to_rc)
    out["row"] = rc.map(lambda t: t[0]).astype(int)
    out["col"] = rc.map(lambda t: t[1]).astype(int)
    if out.duplicated(["plate", "well", "site"]).any():
        raise ValueError("duplicate (plate, well, site) rows in load_data.csv")
    return out.reset_index(drop=True)
