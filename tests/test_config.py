import dataclasses
from pathlib import Path

import pytest

from cellpaint_encoder.config import Config, config_hash, from_dict, load_config, save_config, to_dict

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_and_overrides():
    cfg = load_config(
        None,
        [
            "optim.epochs=7",
            "dino.aug.p_noise=0.5",
            "data.train_sources=[source_2]",
            "data.shard_max_bytes=2.0e6",
        ],
    )
    assert cfg.optim.epochs == 7 and cfg.dino.aug.p_noise == 0.5
    assert cfg.data.train_sources == ("source_2",)
    assert cfg.data.shard_max_bytes == 2e6  # YAML 1.1 reads 2.0e6 as a string: coerced


def test_unknown_key_raises(tmp_path):
    with pytest.raises(KeyError):
        load_config(None, ["optim.nope=1"])
    p = tmp_path / "bad.yaml"
    p.write_text("optim:\n  epochz: 3\n")
    with pytest.raises(KeyError):
        load_config(p)


def test_type_errors():
    with pytest.raises(TypeError):
        load_config(None, ["optim.epochs=abc"])
    with pytest.raises(TypeError):
        load_config(None, ["log.csv=maybe"])


def test_frozen_and_roundtrip(tmp_path):
    cfg = load_config(None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.optim.epochs = 3  # type: ignore[misc]
    save_config(cfg, tmp_path / "c.yaml")
    again = load_config(tmp_path / "c.yaml")
    assert again == cfg
    assert from_dict(to_dict(cfg)) == cfg


def test_hash_ignores_operational_keys():
    a = load_config(None)
    b = load_config(None, ["run.out_dir=elsewhere", "data.num_workers=3", "run.max_steps=5"])
    c = load_config(None, ["optim.base_lr=1e-3"])
    assert config_hash(a) == config_hash(b) != config_hash(c)


@pytest.mark.parametrize("path", sorted((ROOT / "configs").rglob("*.yaml")), ids=lambda p: p.name)
def test_every_config_parses(path):
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    assert cfg.data.heldout_source not in cfg.data.train_sources


def test_pilot_is_the_start_of_the_full_run():
    """Same config hash: the G1 pilot checkpoint resumes straight into the main run."""
    assert config_hash(load_config(ROOT / "configs/pilot.yaml")) == config_hash(
        load_config(ROOT / "configs/full.yaml")
    )
