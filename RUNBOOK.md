# RUNBOOK: from approval to results

This runbook covers the exact setup, commands, cost checkpoints and stop/go gates (DESIGN.md §5.4) for the pilot, the full run and the cut-down run. Every stage is one command, `bash scripts/pod.sh <stage>`, run from the repo root. Nothing here has been run on a GPU yet; §7 lists what remains unverified.

**Order:** G0 (data, CPU pod) → G-B (baselines, dev only) → cache → **G1 pilot (GPU)** → main run → G2 at epoch 25 → G3 → seed 1 + ablations → final scoring (source_6 unlocked once, all methods together).

Before G1: freeze `PREREG.md` (a copy of DESIGN.md §6 plus the evaluation rules fixed in code, listed in §6 below) and tag it `git tag prereg-v1 && git push --tags`. **This has not been done yet.**

---

## 1. Storage: choose one option

### Option A (recommended, DESIGN §1.4): RunPod, tile cache on the pod

- **Pod:**
  - RunPod Secure Cloud, 1× A100 SXM 80 GB ($2.19/h; check the price on launch day);
  - template "RunPod PyTorch" with **Python ≥ 3.11** (the pins need it; `setup` aborts on 3.10: pick a py3.11 image or set `PYTHON=python3.11` after `apt install python3.11-venv`), CUDA 12.x;
  - **≥ 16 vCPU** if offered: the cache build is CPU-bound (§5);
  - container disk 50 GB.
- **Network volume:** 500 GB at $0.07/GB-month (300 GB for the cut-down), mounted at `/workspace`, in the same region as the pod.
- **Clone:** clone the repo into `/workspace/cellpaint-encoder`, so the venv, `data/` (manifest, shards, weights) and `runs/` (checkpoints) survive pod restarts.
- **Shards on the network volume:**
  - The training loader reads samples **by byte offset**: random reads of about 0.65 MB each, about 170 MB/s at 280 samples/s.
  - After `cache`, the `loader` benchmark (`results/g0_loader.json`) shows whether the volume keeps up.
  - If it reports fewer than 400 samples/s with the configured workers, or if the pilot logs `data_wait_frac` > 0.10, copy the shards to local disk and point training at the copy:

    ```
    rsync -a /workspace/cellpaint-encoder/data/shards/ /root/shards/
    ```

    Then add `data.shards_dir=/root/shards/train`. This override is a science key, so it changes the run hash. **Decide before the pilot** and keep the same value afterwards.
  - Local disk needs a container disk of about 400 GB, and it is lost when the pod is terminated. The copy on the volume stays the master.

### Option B (fallback if G0 measures < 100 MB/s from RunPod): AWS us-east-1

- **Instance:**
  - p5.4xlarge spot (1× H100, 3.8 TB NVMe) or g6e.2xlarge (L40S, 450 GiB NVMe);
  - Deep Learning AMI (Ubuntu 22.04);
  - **public subnet** with an **S3 gateway VPC endpoint**. Never route through a NAT gateway: that costs $0.045/GB, about $144 for 3.2 TB.

    ```
    aws ec2 create-vpc-endpoint --vpc-id <vpc> --service-name com.amazonaws.us-east-1.s3 \
        --vpc-endpoint-type Gateway --route-table-ids <rtb-of-the-subnet>
    ```

- **Tile cache on the instance-store NVMe:**

    ```
    sudo mkfs.ext4 /dev/nvme1n1 && sudo mount /dev/nvme1n1 /mnt/nvme && sudo chown $USER /mnt/nvme
    ```

  - Put the repo under `/mnt/nvme`, but keep **`runs/` on an EBS gp3 volume**: instance store is wiped by a spot reclaim or a stop.
  - After a reclaim, rebuild the cache. At the expected ≥ 1 GB/s that takes about 1 h of instance time.
- **Reads:** the bucket is read anonymously in both options, so `--no-sign-request` style access is the default and no AWS credentials are needed for the data.

---

## 2. Setup (once per pod or instance)

```
cd /workspace && git clone <private repo url> cellpaint-encoder && cd cellpaint-encoder
bash scripts/pod.sh setup
```

`setup` does the following:
1. creates `.venv`;
2. installs `torch==2.10.0` from the CUDA 12.8 index (override with `TORCH_INDEX=...cu126` if the driver is older);
3. installs the pinned `requirements.txt`;
4. prints `torch.cuda.is_available()`;
5. runs the test suite (about 2 min);
6. downloads the ImageNet DINO ViT-S/16 weights (86.7 MB) and verifies their SHA-256 (`1566d504…7eb7`, the file checked during development);
7. checks that the ShapeShift anchor tables are present.

**2.3 Anchor tables.** These are made on the laptop and copied to the pod; they are not in git.

```
# laptop
python scripts/import_anchor_wells.py --shapeshift-step2 <path to shapeshift/step2>
rsync -a data/anchor/ <pod>:/workspace/cellpaint-encoder/data/anchor/
```

**Optional E3/E4 sets.** Fetch the JUMP metadata `plate.csv.gz` and `well.csv.gz` from `jump-cellpainting/datasets` into `data/jump/`. Then add `data.plate_table=data/jump/plate.csv.gz data.jump_wells=data/jump/well.csv.gz` to the `g0` stage, or set them in `configs/full.yaml` **before** the manifest is built.

---

## 3. Stages, commands, gates and cost checkpoints

Costs are the planning numbers of DESIGN §5.3 at $2.19/h, given as **cumulative** spend. Stop and reassess if the actual spend passes a checkpoint by more than 20%.

| Stage | Command | Hardware | Pass criteria / decision | Cumulative $ (full) |
|---|---|---|---|---|
| G0 data | `bash scripts/pod.sh g0` | RunPod **CPU** pod, 8–16 vCPU | (1) `results/g0_s3.json` MB_per_s ≥ 150 over 600 s; under 100 → Option B. (2) `results/g0_cache.json` sites_per_s ≥ 40. (3) `build_manifest` raises if any stain is missing in any plate (validates the channel map for every source used). (4) illum dtypes and shapes are printed in `data/write_shards_summary.json` after `cache`. (5) pixel size: see §4 | ≤ 2 |
| Cache | `bash scripts/pod.sh cache` | same CPU pod, or the GPU pod while it is idle | `data/shards/train/stats.json`: `tiles_written` ≈ 357k (the cell filter may drop a few %), `zstd_ratio` ≤ 0.7; `results/g0_loader.json` samples_per_s ≥ 400 | ≤ 13 (CPU pod) |
| G-B baselines | `bash scripts/pod.sh baselines`; CellProfiler: `python scripts/evaluate.py --config configs/full.yaml --cellprofiler-gallery --out results/dev_cp` (CPU, 6.1 GB of profiles); DeepCellMorpho/OpenPhenom: §4 | GPU pod, or CPU for CP | the harness reproduces ShapeShift's CellProfiler replicate AUC within ±0.02 on the same wells; the §2 fidelity gates pass, or the failure is flagged. **source_6 stays locked** (no `eval.unlock_heldout`) | ≤ 24 |
| **G1 pilot** | **`bash scripts/pod.sh pilot`** | GPU pod (A100 SXM) | `results/g1_report.txt` (from `scripts/check_gates.py g1`): throughput ≥ 250/s; data_wait < 10%; GPU busy ≥ 85%; no collapse (teacher entropy away from 0 and from log K; feat_std > 0.1× initial); dev cross-source mAP ≥ ImageNet-frozen + 0.01 (one dev evaluation at the end of epoch 1). The report also prints the plan tier implied by the throughput | ≤ 31 |
| H100 benchmark | `python scripts/train.py --config configs/pilot.yaml --run-dir runs/h100_bench --resume none run.max_minutes=30` on an H100 PCIe pod (after `setup`; copy `data/` from the volume); compare `samples_per_s` with the A100 | H100 PCIe pod, 30 min | switch if its $/sample is ≥ 10% lower: (2.89 / sps_H100) ≤ 0.9 × (2.19 / sps_A100) | ≤ 33 |
| Main run, seed 0 | `bash scripts/pod.sh main` (resumes the pilot checkpoint: same config hash) | GPU pod | **G2 at epoch 25:** first score CellProfiler on exactly the dev-cache wells, `python scripts/evaluate.py --config configs/full.yaml --cellprofiler-gallery --restrict-to-dev-cache --out results/dev_cp_devwells`, then `python scripts/check_gates.py g2 --run runs/full_seed0 --cellprofiler-dev results/dev_cp_devwells/results.json`. Fail → stop, keep the checkpoint, skip seed 1 and the ablations, write the evaluation study. **G3 at epoch 100:** dev metrics at epoch 100 ≥ epoch 50 (`runs/full_seed0/dev_log.csv`); else use the best dev checkpoint (the `ckpt-epoch*` files) by dev cross-source mAP | G2: ≤ 55; end: ≤ 118 |
| Seed 1 | `bash scripts/pod.sh seed1` | GPU pod | only if G1 throughput ≥ 250/s (200–250/s: add `optim.epochs=50`) | ≤ 203 |
| Ablations | `bash scripts/pod.sh abl_natural_mix`, `abl_perimage_norm` (builds its own per-image cache first), `abl_scratch` | GPU pod (Community Cloud is fine) | 25 epochs each | ≤ 267 |
| Final scoring | §3.1 | GPU pod, about 4 h | once, all methods, `eval.unlock_heldout=true` | ≤ 276, plus storage ≈ 53 → **≈ 325 + 25 contingency** |

Each training stage is pre-emption safe:
- `train_loop` restarts on exit code 75 (SIGTERM → checkpoint → exit) with `--resume auto`.
- A hard kill loses at most the work since the last checkpoint (`run.ckpt_every_minutes: 30`).
- The resumed run is bitwise identical on CPU (§8). On GPU it is identical up to non-deterministic kernels.

**Checkpoint size: 0.71 GB** (student + teacher + heads + AdamW). Retention is the last 2 plus every 25th epoch, about 4.3 GB per 100-epoch run, so about 15 GB for the whole plan.

### 3.1 Final scoring (once)

```
python scripts/embed.py --config configs/full.yaml --sets E1,E2,E3,E4 --out runs/embeddings/final \
    --encoder ours=runs/full_seed0/teacher_backbone.pt --encoder imagenet_dino --encoder random_init
#   repeat --encoder ours=... for seed 1 and the ablations (one decode pass feeds all of them)
python scripts/evaluate.py --config configs/full.yaml --sets E1 --cellprofiler-gallery \
    --rep ours=runs/embeddings/final/ours/sites.parquet \
    --rep imagenet_dino=runs/embeddings/final/imagenet_dino/sites.parquet \
    --rep random_init=runs/embeddings/final/random_init/sites.parquet \
    --out results/final eval.unlock_heldout=true
```

- **P1:** `results/final/results.json` → `comparisons.P1_vs_cellprofiler` (mean ΔAP, compound-bootstrap 95% CI, one-sided Wilcoxon).
- **P2–P4:** in the same file, with Holm correction across the secondaries.
- **Sensitivity run:** re-run `embed.py` with `--window 768`.
- **Baselines:** DeepCellMorpho and OpenPhenom embeddings come from the sandbox (§4) as `sites.parquet` files and are passed with extra `--rep` arguments.

### 3.2 Cut-down plan

Same stages, with `CFG=configs/cutdown.yaml`. Its data paths are under `data/cutdown/`.

```
CFG=configs/cutdown.yaml bash scripts/pod.sh g0
CFG=configs/cutdown.yaml bash scripts/pod.sh cache
CFG=configs/cutdown.yaml bash scripts/pod.sh baselines
python scripts/train.py --config configs/cutdown.yaml --run-dir runs/cutdown_seed0 run.max_minutes=120 eval.dev_every_epochs=1   # pilot
python scripts/check_gates.py g1 --run runs/cutdown_seed0 --imagenet-dev results/dev_imagenet.json
python scripts/train.py --config configs/cutdown.yaml --run-dir runs/cutdown_seed0                                    # main (resumes)
python scripts/train.py --config configs/cutdown.yaml --run-dir runs/cutdown_seed1 run.seed=1                         # only if G1 >= 250/s
python scripts/train.py --config configs/ablations/abl_natural_mix.yaml --run-dir runs/cutdown_a1 \
    data.train_sources=[source_2,source_3,source_8] data.manifest=data/cutdown/manifest.parquet \
    data.shards_dir=data/cutdown/shards/train data.dev_shards_dir=data/cutdown/shards/dev                             # only if G1 >= 250/s
```

**Cost checkpoints (cut-down, cumulative):**

| After | $ |
|---|---|
| G0 + cache | ≤ 8 |
| Baselines | ≤ 15 |
| Pilot | ≤ 22 |
| Seed 0 | ≤ 61 |
| Seed 1 | ≤ 100 |
| A1 | ≤ 112 |
| Final | ≤ 119 |
| + storage | ≤ 138 |

---

## 4. Sandbox-only baselines and manual G0 items

**DeepCellMorpho** (CC BY-NC-SA, pickles):
1. In a throwaway container with no network after the download, fetch the LFS weights and run:

   ```
   CPE_SANDBOX=1 python scripts/convert_untrusted_checkpoint.py --i-understand-this-unpickles in.pth out.pt
   ```

2. Convert their `.pkl` embeddings and metadata to parquet the same way.
3. Inside the sandbox, run `scripts/embed.py --encoder deepcellmorpho=out.pt:dinov2_vitb16`.
4. Fidelity gate: compare against their shipped `jump_cp_source6` embeddings on about 20 **non-anchor** source_6 wells, and require cosine ≥ 0.98. This compares vectors only, with no metric.
5. If the gate fails, adjust `clip_pct`, `use_illum` and `channel_order` in `embed/baselines/deepcellmorpho.py`.

**OpenPhenom** (non-commercial EULA):
1. In the sandbox, clone the Hugging Face repo and run `--encoder openphenom=<repo dir>`.
2. First reproduce the repo's `test_huggingface_mae.py`.

**Pixel size (R7):** only source_15 records `Metadata_ImageResolutionX`. For every other source, read the TIFF tags of one site:

```
python -c "import sys,io,tifffile; sys.path.insert(0,'src'); from cellpaint_encoder.data.gallery import Gallery; import pandas as pd; m=pd.read_parquet('data/manifest.parquet'); g=Gallery(); [print(s, {t.name:t.value for t in tifffile.TiffFile(io.BytesIO(g.get(d.key_DNA.iloc[0]))).pages[0].tags.values() if t.name in ('XResolution','ResolutionUnit','ImageDescription')}) for s,d in m.groupby('source')]"
```

If the spread is above 15%, set `data.resample_um_per_px` **before the pilot**. Resampling is not implemented yet (`resample_um_per_px` is only a config key); it must be added and tested first.

---

## 5. Throughput: measured on CPU (2026-11-23) and what is still assumed

Measured on the development laptop: Ryzen 7 5800X3D (8 cores, 16 threads), Windows, Python 3.13, CPU torch 2.10. The data were the 12 real smoke sites (8 from source_2, 4 from source_3, which is LZW-compressed).

| Quantity | Measured | Implication for the GPU run |
|---|---|---|
| Training-loader decode (zstd + `.npy`, 0.65 MB per 448² tile, page-cached) | **≈ 300 samples/s per worker** (265 with 0 workers, 650 with 2, 1,215 with 4) | CPU decode is not a bottleneck at 280/s: 2 workers suffice, and the 10 planned workers leave a wide margin. **Disk/volume random-read speed is not measured** (page cache); run the `loader` benchmark on the pod. |
| zstd ratio on real tiles | **0.65** (6 tiles) | Inside the assumed 0.5–0.7. The raw-size storage budget (358 GB) stands; compressed ≈ 232 GB. |
| Otsu foreground rejection | 0 of 6 candidates (median foreground fraction 0.13) | Consistent with "drops a few %"; real rate from `stats.json` after `cache`. |
| Cache-build CPU rate (decode + illum + plate norm + 2 tiles), files in RAM | **14 sites/s per process**; 39 with 4 processes, 47.5 with 8 | G0's ≥ 40 sites/s on **8 vCPU** is marginal: 8 vCPU are usually 4 physical cores, so expect about 30–40/s. At 30/s the 192k sites take about 1.8 h, below the 2–6 h network estimate, so this is acceptable, but use ≥ 16 vCPU if offered. Threads alone plateau at about 2× one core (GIL), which is why `write_shards` uses processes. |
| Same, files read from local disk | 36 sites/s with 8 processes | Windows antivirus scanning; not expected on Linux. |
| Per-site cost breakdown, one core | TIFF decode 48 ms (LZW-heavy), illum 26 ms, normalise 34 ms, tiles 9 ms | Normalisation is now fused with illumination correction, channel by channel (`read_site_uint8`). |
| Full DINO step on CPU (2×224² + 6×96², ViT-S/16, 65,536-dim head, fp32, batch 16) | 0.23 samples/s | Irrelevant for the GPU estimate; it only proves the full-size recipe runs end to end. |
| Compute per sample (analytic: student forward + backward on 8 crops, teacher forward on 2) | **105 GFLOP** | 280 samples/s on an A100 requires 29 TFLOP/s, **9.4% of the bf16 dense peak** (312). DINO's own V100 anchor (106 samples/s) is about 10% of V100 peak. 280/s is therefore plausible, possibly conservative with SDPA and bf16, but **not verified**. |
| Embedding on CPU (ViT-S, 16 tiles of 224² per site) | about 1 site/s per encoder | GPU inference rate is unverified; DESIGN §5.2 plans 5 h and 4 h for the two passes. |

**Unverified until G0/G1:**
- A100 and H100 samples/s; GPU utilisation; bf16 numerics; `torch.compile` (off by default);
- S3 throughput to the pod;
- network-volume random-read rate;
- the cache-build rate on the pod's vCPUs;
- the real foreground drop rate;
- CUDA wheel and driver compatibility (`cu128`).

The planning rates in DESIGN §5.2 (A100 280/s, H100 PCIe 420/s) stay **[E]**.

---

## 6. Evaluation rules fixed in code (to copy into PREREG.md)

- **Primary metric:**
  - query wells: anchor-compound wells of `eval.query_source`;
  - candidates: anchor wells of the reference sources, never the query's own source;
  - positives: same compound; negatives: other compounds;
  - score: AP per query, mean per compound, mean over compounds;
  - strict variant: also drops candidates at the query's well ID.
- **M1:** the entire candidate pool is restricted to other plates **and** other well positions in the same source. **M2:** only the query itself is removed.
- **eps choice:** per representation, over the grid {0.1, 1, 3, 10}, by maximising the dev different-position replicate-vs-other-compound AUC on dev sources only. Ties go to the larger eps. If that AUC is undefined, the middle of the grid is used.
- **Robust z:**
  - reference: the plate's 16 evaluation DMSO wells;
  - MAD floor: 0.01 × the median over plates of the scale (and at least 1e-12);
  - clip at ±50.
- **Whitening:** fitted on the evaluation DMSO wells of all evaluation plates, keeping 99% of the variance.
- **Gradient clipping:** DINO's per-parameter clipping at 3.0 (`optim.clip_mode=per_param`), as in `main_dino.py`. A global-norm option exists.
- **source_6 lock:** all rows of the held-out source are dropped before any statistic unless `eval.unlock_heldout=true`.

---

## 7. Known gaps and decisions

- PREREG.md and the `prereg-v1` tag are not created.
- `resample_um_per_px` is not implemented (§4); a non-null value raises in `write_shards`/`embed`.
- The DeepCellMorpho and OpenPhenom adapters are tested only against stubs of their documented interfaces.
- The gallery EfficientNet folder name is unverified.
- **Never run yet:** Linux-only paths (`os.pread`, fork-started loader workers, a real SIGTERM), CI on Python 3.11 (the first push is its first run), the `cu128` torch 2.10.0 wheel, any CUDA code path.
- **M4 target-annotation mAP** is not wired (needs a Target-2 target table); M4 cross-source compound retrieval is.
- **Dev-cache DMSO reference (needs a decision):** the dev cache keeps 4 DMSO wells per plate (DESIGN §1.3), so in-training dev metrics z-score against 4 wells, while §4.6 uses 16. G1(c) is internally consistent (ImageNet is scored on the same cache by `dev_eval.py`); G2 scores CellProfiler on the dev-cache wells with `--restrict-to-dev-cache`. Raising the dev cache to 16 DMSO wells per plate costs about +12 GB of cache.
- **Scoping choices made in code:** M1-M3 and the ShapeShift hook use `eval.m1_sources` = sources 2, 3, 6, 8 (§4.3); the dev cache uses `data.dev_sources` = 2, 3, 8 (§1.3); the primary references and M6 include source_10.

---

## 8. CPU smoke test (reproduce on the laptop)

```
py -3.13 -I -X utf8 scripts/import_anchor_wells.py --shapeshift-step2 ../shapeshift/step2
py -3.13 -I -X utf8 scripts/build_manifest.py --config configs/smoke_cpu.yaml
py -3.13 -I -X utf8 scripts/write_shards.py --config configs/smoke_cpu.yaml --procs 4
py -3.13 -I -X utf8 scripts/train.py --config configs/smoke_cpu.yaml --run-dir runs/smoke_ref --resume none
#   kill-and-resume check: start the same command with --run-dir runs/smoke_kill, hard-kill it after
#   step ~23, re-run with --resume auto, then compare the ckpt-step000000040.pt files (train.loop.state_equal)
py -3.13 -I -X utf8 scripts/embed.py --config configs/smoke_cpu.yaml --out runs/smoke_emb \
    --encoder ours=runs/smoke_ref/teacher_backbone.pt --encoder imagenet_dino --encoder random_init
py -3.13 -I -X utf8 scripts/evaluate.py --config configs/smoke_cpu.yaml --out runs/smoke_eval \
    --rep ours=runs/smoke_emb/ours/sites.parquet --rep imagenet_dino=runs/smoke_emb/imagenet_dino/sites.parquet \
    --rep random_init=runs/smoke_emb/random_init/sites.parquet --cellprofiler-wells ../shapeshift/step2/morph_wells_high.parquet
```

**Data:** 3 plates (source_2: 1086292082 and 1086292433; source_3: BR5872d3), 12 sites:
- per plate: 1 training well, 2 shared anchor compounds and 1 DMSO well;
- training: 6 tiles of 448².

**Result on 2026-11-23:**
- 40 optimiser steps (micro-batch 2, accumulation 2, drop-path 0.1, 2 loader workers) in 87 s.
- Hard-killed at step 23 and resumed from step 20: the final student, teacher, optimiser and center are **bitwise identical** to the uninterrupted run, as are all 40 log rows.
- Embedding and evaluation run end to end. The metrics are degenerate at this size, as expected: one DMSO well per plate, and the two source_2 plates share a platemap, so M1 has no units.
