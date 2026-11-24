# RUNBOOK: from approval to results

This runbook covers the exact setup, commands, cost checkpoints and stop/go gates (DESIGN.md §5.4) for the pilot, the full run and the cut-down run. Every stage is one command, `bash scripts/pod.sh <stage>`, run from the repo root. Nothing here has been run on a GPU yet; §7 lists what remains unverified.

**Order:** G0 (data, CPU pod) → G-B (baselines, dev only) → cache → **G1 pilot (GPU)** → main run → G2 at epoch 25 → G3 → seed 1 + ablations → final scoring (source_6 unlocked once, all methods together).

Before G1: freeze `PREREG.md` (a copy of DESIGN.md §6 plus the evaluation rules fixed in code, listed in §6 below) and tag it `git tag prereg-v1 && git push --tags`. **Done 2026-11-24:** see [PREREG.md](PREREG.md); run hash of full/pilot `c900494f6d`. Any later change is a deviation (`DEVIATIONS.md`).

---

## 1. Storage: choose one option

### Option V (chosen 2026-11-24): vast.ai, one instance for every stage

Vast.ai has no network volume that moves between hosts, and an instance's disk size is fixed when it is created. Every stage therefore runs on **one** GPU instance, including G0 and the CPU-bound cache build (about 4–6 h of GPU-instance time). Copying the 232 GB cache between hosts instead would be billed at both ends.

**Choosing an offer.** This query is **untested**: the field names come from the vast.ai CLI reference, but it has never been run.

```
pip install vastai && vastai set api-key <key>          # laptop; keys at cloud.vast.ai/manage-keys
vastai create ssh-key ~/.ssh/id_ed25519.pub              # before creating the instance
vastai search offers 'gpu_name in ["A100 SXM4"] num_gpus=1 gpu_ram>=79 cpu_cores_effective>=16 cpu_ram>=64 disk_space>=600 inet_down>=2000 inet_down_cost<=0.003 cuda_vers>=12.8 reliability>=0.98 geolocation in [US]' -o 'dph'
```

Hover over the offer's price to check its three cost lines:
- **GPU $/h (`dph`).** The cost checkpoints in §3 assume $2.19/h; scale them by `dph / 2.19`.
- **Download $/GB (`inet_down_cost`).** Vast bills ingress, and the full plan reads about **5.2 TB** from S3:
  - 1.91 TB for the training cache;
  - 0.11 TB of normalisation sites;
  - 1.91 TB again for the per-image-norm ablation cache;
  - 1.29 TB of evaluation sets E1–E4 at final scoring.

  That is about $16 at $0.003/GB and about $104 at $0.02/GB. The host choice matters.
- **Storage $/GB-month.** It is billed for as long as the instance exists, running or stopped, until it is destroyed.

**Disk: `--disk 600`.**
- Budgeted at raw size, as DESIGN §1.3 does: training cache 358 GB (≈ 232 GB at the measured zstd ratio of 0.65), dev cache ≈ 70 GB, checkpoints ≈ 15 GB, and about 40 GB for the venv, the image, CellProfiler profiles and embeddings. That totals ≈ 480 GB.
- The per-image-norm ablation needs a second training cache (358 GB raw), which does not fit next to the first. Run `abl_perimage_norm` **last**, after deleting `data/shards/train`. `pod.sh` checks free space and stops with instructions instead of deleting anything.
- To keep both caches, use `--disk 900`.

**Rent on demand, not interruptible.** An outbid interruptible instance stops and may stay stopped.

**Image.** `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` (amd64, 4.3 GB). It ships Python 3.11, per PyTorch's `Dockerfile` and `docker.Makefile` at tag v2.7.0. `setup` installs its own pinned torch 2.10.0 (cu128) in a venv, so the image only supplies Python and the CUDA userspace. The host driver must support CUDA 12.8 (`cuda_vers>=12.8`).

```
vastai create instance <OFFER_ID> --image pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime \
    --disk 600 --ssh --direct --label cpe-full --cancel-unavail
vastai ssh-url <INSTANCE_ID>                             # ssh://root@HOST:PORT  ->  ssh -p PORT root@HOST
```

**On the instance:**
```
apt-get update && apt-get install -y git tmux rsync curl
tmux new -s cpe                                          # every stage runs inside tmux; reattach: tmux attach -t cpe
mkdir -p /workspace && cd /workspace
git clone https://github.com/Sam-Wo/cellpaint-encoder && cd cellpaint-encoder
bash scripts/pod.sh setup
```

`pod.sh` sets `PROCS` from the container's cgroup CPU quota, because `nproc` inside a vast container can report every host core, and prints the value. If it differs from the offer's `cpu_cores_effective`, export `PROCS=<cpu_cores_effective>`.

**Anchor tables, from the laptop:**
```
scp -P <PORT> -r data/anchor root@<HOST>:/workspace/cellpaint-encoder/data/
```

**Backups.** The instance disk holds the only copy of everything. After G1, G2 and G3, after every finished run, and every few hours during the main run, copy the run logs, the newest checkpoint (0.71 GB) and `results/` to the laptop:

```
scp -P <PORT> -r root@<HOST>:/workspace/cellpaint-encoder/results ./results_vast
```

**Stopping and destroying.**
- **Do not stop the instance between stages.** Storage keeps billing while it is stopped. Whether the GPU can be rented by someone else meanwhile, so that the instance cannot restart, is not documented; treat it as a risk.
- When everything is copied off, destroy the instance: `vastai destroy instance <INSTANCE_ID>`.

**G0 failure on vast** (S3 < 100 MB/s): destroy the instance, which holds nothing yet, and try another US host. Fall back to Option B (AWS) only if two hosts fail.

**H100 benchmark (§3).** Skip it on vast unless G1 measures < 250 samples/s. It needs a second instance and a billed copy of the cache.

### Option A (DESIGN §1.4, not used): RunPod, tile cache on the pod

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
cd /workspace && git clone https://github.com/Sam-Wo/cellpaint-encoder && cd cellpaint-encoder
bash scripts/pod.sh setup
```

On vast.ai, first run the apt and tmux lines of §1 Option V.

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
| G0 data | `bash scripts/pod.sh g0` | RunPod **CPU** pod, 8–16 vCPU (vast.ai: the GPU instance itself, ≥ 16 vCPU) | (1) `results/g0_s3.json` MB_per_s ≥ 150 over 600 s; under 100 → Option B (vast.ai: another host first, see §1 Option V). (2) `results/g0_cache.json` sites_per_s ≥ 40. (3) `build_manifest` raises if any stain is missing in any plate (validates the channel map for every source used). (4) illum dtypes and shapes are printed in `data/write_shards_summary.json` after `cache`. (5) pixel size: see §4 | ≤ 2 |
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

**Pixel size (R7): done, so resampling is on.** TIFF resolution tags cannot be used: they hold writer defaults or the sensor pixel size. The authoritative values come from jump-cellpainting/datasets `metadata/microscope_config.csv` at commit `b3b4555d44cac440ffda7e25f3adbeb34fbee7fe`, column `Metadata_Pixel_Size_Microns`. These values are in µm/px; ✱ marks the sources we use.

| source | 1 | 2✱ | 3✱ | 4 | 5 | 6✱ (held out) | 7 | 8✱ | 9 | 10✱ | 11 | 13 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| µm/px | 0.597976 | 0.65 | 0.597976 | 0.597976 | 0.64974 | 0.646187 | 0.65 | 0.7032 | 0.597976 | 0.6516944 | 0.597976 | 0.65 | 0.597976 |

Two values were checked independently:
- source_8 matches its MetaSeries TIFF description (spatial-calibration-x = 0.7032 µm, binning 2×2).
- source_3's 0.598 matches source_15's value as verified in its `load_data.csv`.

Across the used sources the spread is 0.598 to 0.703, about 17.6%. That is above the 15% threshold, so the pre-registered rule applies: resample to a common µm/px at cache time, decided before the pilot.

**Settings.** Identical in every config, including the pilot, the cut-down plan, the ablations and the smoke config. They are science keys and part of the run hash.
- Target: `data.resample_um_per_px: 0.65`.
- Tolerance: `data.resample_tol: 0.02`.
- Table: `data.pixel_size_um` holds all 13 rows above.
- Scale factor = native µm/px ÷ 0.65, and the new size = round(old size × factor).

**Which sources are resampled:**
- Any source with |factor − 1| ≤ 0.02 keeps its native pixels, bit for bit. That covers source_2, source_10, the held-out source_6 (so the DeepCellMorpho fidelity gate stays valid), and source_7.
- source_3 is resampled ×0.920, down: a 1080² site becomes 994².
- source_8 is resampled ×1.082, up.
- If resampling is on and a source is missing from the table, the run raises. This covers any optional E3/E4 plates.

**Order and interpolation** (`src/cellpaint_encoder/data/resample.py`):
1. Illumination correction and the per-plate (or per-image) normalisation's affine map are applied at native resolution. The plate percentiles are still computed at native resolution.
2. The float image is resampled per channel with `torch.nn.functional.interpolate(mode="bilinear", antialias=True, align_corners=False)` on CPU float32. The same call handles both directions.
3. The image is then clipped, rounded, tiled and foreground-filtered, and the embedding crop windows are taken.

One function (`resample.resample_site`) is used by every pixel path:
- the training cache, the dev cache and the per-image ablation cache (`write_shards`);
- `embed.py` for every encoder: ours, ImageNet and random through `normalise_site`, and the DeepCellMorpho and OpenPhenom adapters. These two normalise per crop or per image, so they resample their float input before the crop grid.

**Guard against a train/eval mismatch:**
- Both caches' `stats.json` and each `embed_info.json` record `resample_um_per_px`, `resample_tol` and the applied per-source scales.
- `train.py`, the dev hook, `dev_eval.py` and `embed.py` refuse shards, or an `ours=` checkpoint config, whose settings differ from the config.
- Shards written before this change count as "not resampled", so they are refused and must be rebuilt.

---

## 5. Throughput: measured on CPU (2026-11-23) and what is still assumed

**Stale since R7:** these cache-build rates were measured before resampling was added (source_3 ×0.92, source_8 ×1.08 per site). `bench_io.py cache` now includes it, so G0's `results/g0_cache.json` on the instance is the number to trust.

Measured on the development laptop: Ryzen 7 5800X3D (8 cores, 16 threads), Windows, Python 3.13, CPU torch 2.10. The data were the 12 real smoke sites (8 from source_2, 4 from source_3, which is LZW-compressed).

**Caveat:** an unrelated CPU-heavy job (15 worker processes) started on the same machine at 10:17, part-way through these measurements. The loader and threaded cache numbers were taken before it. The multi-process cache numbers and the CPU step number may be depressed by it, so the multi-process scaling below is a **lower bound**. Re-run `bench_io.py cache` on the pod in any case.

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

## 6. Evaluation rules fixed in code (copied into PREREG.md §4)

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

- The DeepCellMorpho and OpenPhenom adapters are tested only against stubs of their documented interfaces.
- The gallery EfficientNet folder name is unverified.
- **Never run yet:** Linux-only paths (`os.pread`, fork-started loader workers, a real SIGTERM), CI on Python 3.11 (the first push is its first run), the `cu128` torch 2.10.0 wheel, any CUDA code path.
- **M4:** both halves are wired. M4b (target annotation) uses `eval.target_table=data/jump/target2_targets.csv`, written by `scripts/fetch_jump_metadata.py` during `pod.sh setup`; definition in PREREG §3. E3/E4 are on in `full.yaml`.
- **Dev-cache DMSO reference (decided in PREREG §5.5: keep 4):** the dev cache keeps 4 DMSO wells per plate (DESIGN §1.3), so in-training dev metrics z-score against 4 wells, while §4.6 uses 16. G1(c) is internally consistent (ImageNet is scored on the same cache by `dev_eval.py`); G2 scores CellProfiler on the dev-cache wells with `--restrict-to-dev-cache`. Raising the dev cache to 16 DMSO wells per plate costs about +12 GB of cache.
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

**Result on 2026-11-23, re-run from the final commit:**
- No network bytes were used: the byte cap was set to 1 and everything came from the local mirror.
- 40 optimiser steps (micro-batch 2, accumulation 2, drop-path 0.1, 2 loader workers) in 189 s, on a CPU shared with another job; an unloaded earlier run took 87 s.
- Hard-killed (`taskkill /F /T`) after logged step 23 and resumed with `--resume auto` from `ckpt-step20`: the final student, teacher, optimiser and DINO center are **bitwise identical** to the uninterrupted run, as are all 40 log rows (loss, grad norm, entropies, lr).
- Embedding and evaluation run end to end. The metrics are degenerate at this size, as expected: one DMSO well per plate, and the two source_2 plates share a platemap, so M1 has no units.
