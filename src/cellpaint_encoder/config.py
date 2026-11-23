"""YAML -> nested frozen dataclasses, ``key.sub=value`` overrides, strict keys, config hash.

DESIGN.md 3.3: unknown keys raise; the resolved config is saved in the run directory and its
hash names the run.

>>> cfg = load_config(None, ["optim.epochs=3", "run.seed=7"])
>>> cfg.optim.epochs, cfg.run.seed
(3, 7)
>>> load_config(None, ["optim.epochz=3"])
Traceback (most recent call last):
...
KeyError: "unknown config key 'optim.epochz'"
>>> len(config_hash(cfg))
10
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class DataCfg:
    # --- what the manifest selects (DESIGN.md 1.2) ---
    anchor_wells: str = "data/anchor/anchor_wells_high.parquet"  # source,batch,plate,well,compound_id,is_dmso
    jump_wells: str | None = None  # optional JUMP well.csv.gz: DMSO identity of non-anchor wells
    train_sources: tuple[str, ...] = ("source_2", "source_3", "source_8", "source_10")
    heldout_source: str = "source_6"  # never trained on; metrics locked (4.1)
    eval_sources: tuple[str, ...] = ("source_2", "source_3", "source_8", "source_10", "source_6")
    dev_sources: tuple[str, ...] = ("source_2", "source_3", "source_8")  # dev cache (1.3)
    excluded_sources: tuple[str, ...] = ("source_15", "source_7")  # 15: no RNA; 7: eval-only (dose)
    plates: tuple[str, ...] = ()  # explicit plate list (smoke); empty = every anchor-hosting plate
    manifest_seed: int = 0  # fixes plate split, site choice, DMSO subsets (same for every training seed)
    heldout_plate_frac: float = 0.2  # per source, stratified by batch
    sites_per_well: int = 3  # training sites per well (fixed by seed)
    n_eval_dmso_per_plate: int = 16  # masked from training, plate z-score reference
    n_dev_dmso_per_plate: int = 4  # subset of the 16, in the dev cache
    norm_sites_per_plate: int = 64  # DMSO sites used for the per-plate percentiles
    eval_sites_per_well: int = 0  # 0 = all sites
    # smoke-size limits (0 = no limit)
    max_train_wells_per_plate: int = 0
    max_anchor_wells_per_plate: int = 0
    eval_compounds: tuple[str, ...] = ()  # restrict anchor compounds (smoke)
    # extra evaluation-only sets (1.2): E2 source_7 low dose, E3 Target-2, E4 all-DMSO plates
    anchor_wells_low: str | None = None  # same schema as anchor_wells, 0.625 uM (source_7)
    plate_table: str | None = None  # JUMP metadata/plate.csv.gz (Metadata_PlateType)
    e3_sources: tuple[str, ...] = ("source_2", "source_3", "source_6", "source_8", "source_10")
    e3_plates_per_source: int = 4
    e4_sources: tuple[str, ...] = ("source_3", "source_10")
    e4_plates_per_source: int = 4
    extra_sites_per_well: int = 4
    # --- cache build (1.3) ---
    raw_mirror: str | None = None  # local mirror of the bucket (keys under it); None = no mirror
    byte_cap: float = 0  # hard cap on new downloads in bytes (0 = none)
    s3_workers: int = 64
    norm_mode: str = "per_plate_dmso"  # | per_image (ablation A2)
    pct_lo: float = 0.1
    pct_hi: float = 99.9
    tile_size: int = 448
    tiles_per_site: int = 2
    fg_min: float = 0.01  # Otsu DNA foreground fraction to keep a tile
    dev_site_size: int = 896  # dev cache stores centred 896^2 sites (4x4 grid of 224)
    shard_max_bytes: float = 1.0e9
    zstd_level: int = 3
    resample_um_per_px: float | None = None  # R7: resample to a common pixel size if set
    # --- training-time loading (3.4) ---
    shards_dir: str = "data/shards/train"
    dev_shards_dir: str = "data/shards/dev"
    manifest: str = "data/manifest.parquet"
    balance: str = "source"  # source | natural (ablation A1)
    dmso_cap: float = 0.2
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    tile_cache_mb: int = 0  # optional in-RAM LRU of decoded tiles per worker (0 = off)
    channel_mean: tuple[float, ...] = ()  # empty = read from shards stats.json
    channel_std: tuple[float, ...] = ()


@dataclass(frozen=True)
class ModelCfg:
    arch: str = "vit_small"
    patch_size: int = 16
    in_chans: int = 5
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_path: float = 0.1
    init: str = "imagenet_dino"  # imagenet_dino | random (ablation A3)
    init_weights: str = "data/weights/dino_deitsmall16_pretrain.pth"
    compile: bool = False


@dataclass(frozen=True)
class AugCfg:
    gain: tuple[float, float] = (0.8, 1.25)
    offset_std: float = 0.05
    gamma: tuple[float, float] = (0.8, 1.25)
    p_intensity: float = 0.5
    blur_sigma: tuple[float, float] = (0.1, 1.0)
    p_blur_global1: float = 1.0
    p_blur_global2: float = 0.1
    p_blur_local: float = 0.5
    noise_std: float = 0.02
    p_noise: float = 0.3
    p_channel_dropout: float = 0.1


@dataclass(frozen=True)
class DinoCfg:
    out_dim: int = 65536
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    norm_last_layer: bool = True
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: float = 30
    student_temp: float = 0.1
    center_momentum: float = 0.9
    momentum_teacher: float = 0.996
    freeze_last_layer_epochs: float = 1
    global_crop_size: int = 224
    local_crop_size: int = 96
    n_local_crops: int = 6
    scale_jitter: float = 0.2
    aug: AugCfg = field(default_factory=AugCfg)


@dataclass(frozen=True)
class OptimCfg:
    epochs: int = 100
    batch_size: int = 256  # per optimiser step (effective batch)
    accum_steps: int = 1
    base_lr: float = 5e-4  # scaled by batch/256
    min_lr: float = 1e-6
    warmup_epochs: float = 10
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    clip_mode: str = "per_param"  # per_param (DINO's clip_gradients) | global
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(frozen=True)
class RunCfg:
    name: str = "run"
    out_dir: str = "runs"
    seed: int = 0
    device: str = "auto"  # auto | cpu | cuda
    precision: str = "bf16"  # bf16 | fp16 (GradScaler) | fp32
    epoch_size: int = 0  # tiles per epoch; 0 = number of tiles in the shards
    max_steps: int = 0  # stop (with a checkpoint) after this many optimiser steps; schedule unchanged
    max_minutes: float = 0  # wall-clock stop (pilot); schedule unchanged
    ckpt_every_minutes: float = 30
    ckpt_every_steps: int = 0
    keep_last: int = 2
    keep_every_epochs: int = 25
    log_every_steps: int = 50
    threads: int = 0  # torch intra-op threads (0 = default)
    deterministic: bool = False  # cudnn.benchmark off + deterministic algorithms
    cudnn_benchmark: bool = True


@dataclass(frozen=True)
class LogCfg:
    csv: bool = True
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "cellpaint-encoder"


@dataclass(frozen=True)
class EvalCfg:
    dev_every_epochs: int = 10
    grid: int = 4  # 4x4 of 224 (4.6)
    crop: int = 224
    batch_size: int = 64
    dev_query_sources: tuple[str, ...] = ("source_2", "source_3", "source_8")
    dev_reference_sources: tuple[str, ...] = ("source_2", "source_3", "source_8")
    reference_sources: tuple[str, ...] = ("source_2", "source_3", "source_8", "source_10")
    m1_sources: tuple[str, ...] = ("source_2", "source_3", "source_6", "source_8")  # M1-M3 (4.3)
    low_dose_source: str = "source_7"  # M5 queries (E2)
    query_source: str = "source_6"
    unlock_heldout: bool = False  # source_6 metrics stay locked until the final scoring
    eps_grid: tuple[float, ...] = (0.1, 1.0, 3.0, 10.0)
    var_keep: float = 0.99
    clip_z: float = 50.0
    mad_floor_rel: float = 0.01
    mad_floor_abs: float = 1e-12
    n_boot: int = 10000
    n_null: int = 1000
    moa_table: str | None = None  # csv with compound_id, moa
    seed: int = 0


@dataclass(frozen=True)
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    dino: DinoCfg = field(default_factory=DinoCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    run: RunCfg = field(default_factory=RunCfg)
    log: LogCfg = field(default_factory=LogCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)


# keys that do not change the science; excluded from the hash so that moving a run directory,
# changing loader workers or log cadence does not rename the run
NON_SCIENCE_KEYS = {
    "run.out_dir",
    "run.name",
    "run.max_steps",
    "run.max_minutes",
    "run.ckpt_every_minutes",
    "run.ckpt_every_steps",
    "run.keep_last",
    "run.keep_every_epochs",
    "run.log_every_steps",
    "run.threads",
    "data.num_workers",
    "data.prefetch_factor",
    "data.pin_memory",
    "data.tile_cache_mb",
    "data.raw_mirror",
    "data.byte_cap",
    "data.s3_workers",
    "log.csv",
    "log.tensorboard",
    "log.wandb",
    "log.wandb_project",
    "eval.batch_size",
    "eval.dev_every_epochs",  # so the G1 pilot is the first 2 h of the main run (same hash)
}


def _coerce(tp: Any, value: Any, key: str) -> Any:
    origin = get_origin(tp)
    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise TypeError(f"{key}: expected a mapping")
        return _build(tp, value, key + ".")
    if origin is tuple:
        args = get_args(tp)
        if value is None:
            value = ()
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{key}: expected a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, key) for v in value)
        if len(args) != len(value):
            raise TypeError(f"{key}: expected {len(args)} values")
        return tuple(_coerce(a, v, key) for a, v in zip(args, value))
    if origin is not None and type(None) in get_args(tp):  # Optional[X] / X | None
        if value is None:
            return None
        inner = [a for a in get_args(tp) if a is not type(None)][0]
        return _coerce(inner, value, key)
    if tp is float:
        if isinstance(value, str):  # YAML 1.1 reads "2.0e6" (no exponent sign) as a string
            try:
                return float(value)
            except ValueError:
                raise TypeError(f"{key}: expected float, got {value!r}") from None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise TypeError(f"{key}: expected float, got {value!r}")
    if tp is int and isinstance(value, float) and value.is_integer():
        return int(value)
    if tp in (int, str, bool) and not isinstance(value, tp):
        raise TypeError(f"{key}: expected {tp.__name__}, got {value!r}")
    if tp is int and isinstance(value, bool):
        raise TypeError(f"{key}: expected int, got bool")
    return value


def _build(cls, d: dict, prefix: str = ""):
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    for k in d:
        if k not in names:
            raise KeyError(f"unknown config key '{prefix}{k}'")
    kw = {}
    for f in fields(cls):
        if f.name in d:
            kw[f.name] = _coerce(hints[f.name], d[f.name], prefix + f.name)
    return cls(**kw)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = d.pop("base", None)
    if base:
        d = _deep_merge(_read_yaml((path.parent / base).resolve()), d)
    return d


def _set_path(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
        if not isinstance(cur, dict):
            raise KeyError(f"cannot override inside non-mapping '{dotted}'")
    cur[parts[-1]] = value


def _check_path(dotted: str) -> None:
    cls: Any = Config
    for p in dotted.split("."):
        if not is_dataclass(cls) or p not in {f.name for f in fields(cls)}:
            raise KeyError(f"unknown config key '{dotted}'")
        cls = get_type_hints(cls)[p]


def load_config(path: str | Path | None, overrides: list[str] | tuple[str, ...] = ()) -> Config:
    d: dict = _read_yaml(Path(path)) if path else {}
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value: {ov!r}")
        k, v = ov.split("=", 1)
        _check_path(k.strip())
        _set_path(d, k.strip(), yaml.safe_load(v))
    return _build(Config, d)


def to_dict(cfg: Config) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(cfg)))


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def config_hash(cfg: Config) -> str:
    flat = {k: v for k, v in _flatten(to_dict(cfg)).items() if k not in NON_SCIENCE_KEYS}
    return hashlib.sha256(json.dumps(flat, sort_keys=True).encode()).hexdigest()[:10]


def save_config(cfg: Config, path: Path) -> None:
    path.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False), encoding="utf-8")


def from_dict(d: dict) -> Config:
    return _build(Config, d)
