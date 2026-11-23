"""Convert pickle-based third-party files ONCE, INSIDE A THROWAWAY CONTAINER (DESIGN.md 2, "Security").

DeepCellMorpho ships ``.pth`` checkpoints and ``.pkl`` embeddings/metadata. Unpickling executes
arbitrary code, so this script refuses to run unless ``CPE_SANDBOX=1`` is set and
``--i-understand-this-unpickles`` is passed. It writes:

* ``.pth``  -> ``<out>.pt``: a plain ``{name: tensor}`` dict, loadable with ``torch.load(weights_only=True)``
* ``.pkl``  -> ``<out>.parquet`` (DataFrames, or dicts of equal-length arrays)

    docker run --rm --network none -v $PWD/sandbox:/w -e CPE_SANDBOX=1 python:3.11 \
        bash -c "pip install torch pandas pyarrow && python /w/convert_untrusted_checkpoint.py \
                 --i-understand-this-unpickles /w/in/dinov2_vitb16.pth /w/out/dcm_dinov2_vitb16.pt"

Copy only the converted outputs back. Never redistribute them (CC BY-NC-SA 4.0).
"""

import argparse
import os
import pickle
import sys
from pathlib import Path


def flatten_tensors(obj, prefix="", out=None):
    import torch

    out = {} if out is None else out
    if isinstance(obj, torch.Tensor):
        out[prefix.rstrip(".")] = obj.detach().cpu().contiguous()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten_tensors(v, f"{prefix}{k}.", out)
    elif hasattr(obj, "state_dict"):
        flatten_tensors(obj.state_dict(), prefix, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-understand-this-unpickles", action="store_true")
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    if os.environ.get("CPE_SANDBOX") != "1" or not a.i_understand_this_unpickles:
        sys.exit("refusing: run only inside a throwaway container with CPE_SANDBOX=1 and the explicit flag")
    src, dst = Path(a.src), Path(a.dst)
    if src.suffix in (".pth", ".pt", ".ckpt"):
        import torch

        obj = torch.load(src, map_location="cpu", weights_only=False)  # untrusted: sandbox only
        sd = flatten_tensors(obj)
        torch.save(sd, dst)
        torch.load(dst, weights_only=True)  # verify the output is safe to load
        print(f"wrote {dst} with {len(sd)} tensors")
    elif src.suffix == ".pkl":
        import pandas as pd

        with open(src, "rb") as f:
            obj = pickle.load(f)  # untrusted: sandbox only
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        for c in df.columns:
            if (
                df[c].dtype == object
                and len(df)
                and hasattr(df[c].iloc[0], "__len__")
                and not isinstance(df[c].iloc[0], str)
            ):
                df[c] = df[c].map(lambda v: list(map(float, v)))
        df.columns = [str(c) for c in df.columns]
        df.to_parquet(dst, index=False)
        print(f"wrote {dst}: {df.shape}")
    else:
        sys.exit(f"unsupported file type {src.suffix}")


if __name__ == "__main__":
    main()
