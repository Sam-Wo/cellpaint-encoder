# cellpaint-encoder

Self-supervised (DINO) ViT-S/16 encoder for 5-channel Cell Painting images from JUMP-CP, trained with a hand-written PyTorch loop and evaluated with the batch- and well-position controls learned in [ShapeShift](https://github.com/Sam-Wo/shapeshift).

**Status:** design pre-registered in [DESIGN.md](DESIGN.md) (first commit); analysis plan frozen in [PREREG.md](PREREG.md) (tag `prereg-v1`) before the GPU pilot. The code is complete and tested on CPU, including a smoke run on a handful of real JUMP sites. **No GPU training has been run.** [RUNBOOK.md](RUNBOOK.md) gives the exact commands for the gates, the pilot, the full run and the cut-down run.

## Layout

The layout follows DESIGN.md §7. The configs are flat (`configs/{full,cutdown,pilot,smoke_cpu}.yaml` plus `configs/ablations/`), and `scripts/write_shards.py` is the cache builder that §7 calls `build_cache.py`.

| Path | Contents |
|---|---|
| `src/cellpaint_encoder/data/` | anonymous S3 access, `load_data.csv` stain mapping, illumination correction, per-plate DMSO normalisation, tiles, WebDataset shards, source-balanced sampler, GPU multi-crop augmentation, split manifest |
| `src/cellpaint_encoder/models/` | ViT-S/16 with a 5-channel patch embedding inflated from ImageNet DINO, DINO head, DINO loss |
| `src/cellpaint_encoder/train/` | hand-written loop (bf16, accumulation, EMA, schedules, clipping), exact checkpoint/resume, logging |
| `src/cellpaint_encoder/embed/` | one decode feeding many encoders; per-model crop grids and preprocessing adapters (ours, ImageNet, random, DeepCellMorpho, OpenPhenom, gallery CP-CNN) |
| `src/cellpaint_encoder/eval/` | aggregation, robust z and whitening, copairs-style mAP (primary cross-source, M1–M6), probes, statistics, CellProfiler baseline, ShapeShift hook |
| `scripts/` | `build_manifest`, `write_shards`, `train`, `embed`, `evaluate`, `bench_io`, `import_anchor_wells`, `convert_untrusted_checkpoint` |
| `tests/` | fast CPU tests on synthetic data; fixtures are real `load_data.csv` header rows |

## Quick start (CPU)

```
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
```

The smoke test on real data is described in RUNBOOK.md §8.

## Licences

- Code: see `LICENSE`. DESIGN.md §2 plans Apache-2.0, but the file currently says MIT; this is to be decided before the repo is made public.
- Trained weights: CC-BY-4.0. The data is CC0 (Cell Painting Gallery); the initialisation is ImageNet DINO ViT-S/16 (Apache-2.0).
- DeepCellMorpho (CC BY-NC-SA 4.0) and OpenPhenom (Recursion non-commercial EULA) weights and embeddings are **never redistributed**. Only metrics derived from them are published. Their adapters load converted weights only, and only in a sandbox.
