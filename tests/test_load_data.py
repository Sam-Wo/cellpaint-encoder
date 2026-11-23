"""Channel mapping goes through URL_Orig<Stain> columns only (DESIGN.md 1.1). Fixtures are the real
header + first rows of load_data.csv from six sources."""

import pytest

from cellpaint_encoder import CHANNELS
from cellpaint_encoder.data.load_data import MissingChannelError, parse_load_data, well_to_rc
from conftest import FIXTURES


def load(src):
    return parse_load_data((FIXTURES / f"load_data_{src}.csv").read_bytes())


# expected file-name channel token per stain, per source (from the headers; differs by source)
EXPECTED = {
    "source_2": {"DNA": "C01", "ER": "C02", "RNA": "C03", "AGP": "C04", "Mito": "C05"},
    "source_3": {"Mito": "ch1", "AGP": "ch2", "RNA": "ch3", "ER": "ch4", "DNA": "ch5"},
    "source_7": {"DNA": "C01", "Mito": "C02", "RNA": "C03", "ER": "C04", "AGP": "C05"},
    "source_8": {"Mito": "_w1", "AGP": "_w2", "RNA": "_w3", "ER": "_w4", "DNA": "_w5"},
    "source_10": {"DNA": "C01", "ER": "C02", "RNA": "C03", "AGP": "C04", "Mito": "C05"},
}


@pytest.mark.parametrize("src", sorted(EXPECTED))
def test_channel_mapping(src):
    d = load(src)
    assert (d.source == src).all()
    for stain, token in EXPECTED[src].items():
        name = d[f"key_{stain}"].iloc[0].rsplit("/", 1)[-1]
        assert token in name, (src, stain, name)
    assert [c for c in d.columns if c.startswith("key_")] == [f"key_{c}" for c in CHANNELS]
    assert not any("Brightfield" in c for c in d.columns)  # brightfield dropped


def test_mito_differs_by_source():
    tokens = {
        s: load(s)["key_Mito"].iloc[0].rsplit("/", 1)[-1] for s in ("source_3", "source_7", "source_10")
    }
    assert "ch1" in tokens["source_3"] and "C02" in tokens["source_7"] and "C05" in tokens["source_10"]


def test_source15_missing_rna_raises():
    with pytest.raises(MissingChannelError, match="RNA"):
        load("source_15")


def test_keys_are_bucket_keys_and_spaces_kept():
    d = load("source_3")
    k = d.key_DNA.iloc[0]
    assert k.startswith("cpg0016-jump/source_3/") and "Measurement 1" in k


def test_well_to_rc():
    assert well_to_rc("A01") == (0, 0)
    assert well_to_rc("P24") == (15, 23)
    assert well_to_rc("AA01") == (26, 0)
    with pytest.raises(ValueError):
        well_to_rc("1A")
