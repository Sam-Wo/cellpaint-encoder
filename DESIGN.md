# cellpaint-encoder: design doc

*Draft v1, 2026-11-22. Design only: no code, no GPU rental, no image downloads.*

This project trains a 5-channel Cell Painting image encoder with self-supervision on JUMP-CP compound images, then compares it with CellProfiler and with released pretrained encoders. The evaluation is designed around the confounds found in ShapeShift (well position, DMSO-null saturation, plate/source/batch effects, and the source_7 dose difference).

**Evidence labels.** Every number in this doc carries one of three labels:
- **[V]** verified, with a link or the exact listing/request used;
- **[D]** derived arithmetically from [V] numbers;
- **[E]** an estimate or assumption.

All S3 facts were checked on 2026-11-22 with anonymous REST listings of `https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&prefix=...`, range reads of TIFF headers (8 bytes plus an 8 KB IFD), and range reads of `load_data.csv` headers. No image was downloaded. Appendix A lists what was fetched.

---

## 0. Summary

| | Full plan | Cut-down |
|---|---|---|
| Training sources | source_2, source_3, source_8, source_10 | source_2, source_3, source_8 |
| Held-out source | source_6 (never trained on) | source_6 |
| Extra evaluation-only sets | source_7 (0.625 µM), 20 JUMP Target-2 plates, 8 all-DMSO plates | source_7, 2 all-DMSO plates |
| Training images | 178,566 sites (3 per well), 1.91 TB streamed once [D] | 101,446 sites (2 per well), 1.10 TB [D] |
| Evaluation images | 113,295 sites, 1.29 TB [D] | 70,233 sites, 0.84 TB [D] |
| Model | ViT-S/16 with 5-channel patch embedding, DINO, initialised from ImageNet DINO ViT-S/16 | same |
| Schedule | 100 epochs × 357,132 tiles, 2 seeds, plus 3 ablations at 25 epochs | 80 epochs × 202,892 tiles, 2 seeds, 1 ablation (seed 2 and the ablation are dropped if G1 measures under 250/s) |
| GPU | RunPod Secure A100 SXM 80 GB at $2.19/h [V]; benchmark H100 PCIe at $2.89/h [V] in the pilot | same |
| Expected cost | **≈ $325 + $25 contingency = $350** at the planning rate of 280 samples/s [E] | **≈ $138** at 280 samples/s; **≈ $126** at half that rate with seed 2 and the ablation dropped [D from E] |
| Egress cost | $0: the bucket accepts anonymous, unsigned requests, which cannot be billed to the requester [V] | $0 |
| Primary pre-registered test | Cross-source replicate retrieval: source_6 query wells against training-source references, with other-compound negatives. Ours must beat CellProfiler on identical wells and aggregation, with the compound-bootstrap 95% CI excluding 0. | same, with one seed |

---

## 1. Data

### 1.1 Storage format (verified)

**Layout.** cpg0016 lives in bucket `cellpainting-gallery` (us-east-1) under `cpg0016-jump/` [V]. Each source has `images/`, `workspace/`, and, for most sources, `workspace_dl/` [V].
- Images: `cpg0016-jump/{source}/images/{batch}/images/{plate_dir}/...`, with illumination functions at `.../images/{batch}/illum/{plate_barcode}/{barcode}_Illum{DNA,ER,RNA,AGP,Mito}.npy` [V].
- The per-site channel manifest is `cpg0016-jump/{source}/workspace/load_data_csv/{batch}/{barcode}/load_data.csv`, plus `load_data_with_illum.csv`. Its columns include `URL_OrigDNA`, `URL_OrigER`, `URL_OrigRNA`, `URL_OrigAGP`, `URL_OrigMito`, often `URL_OrigBrightfield{,_L,_H}`, and `Metadata_{Source,Batch,Plate,Well,Site}` [V].
- Example listing: <https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_2/images/20210614_Batch_1/images/1053600674/>
- Example manifest: <https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_2/workspace/load_data_csv/20210614_Batch_1/1053600674/load_data_with_illum.csv>

**Format.** One single-page, 16-bit greyscale TIFF per channel per site. There is no multi-channel file. Some sources write uncompressed TIFFs and others LZW-compressed ones.

| Source | File naming | Site size (px) | Compression | Sites/well | Fluorescence channels | Brightfield | Bytes/site, 5 fluorescence channels |
|---|---|---|---|---|---|---|---|
| source_2 | `{plate}_{well}_T0001F001L01A01Z01C01.tif` (Yokogawa-style) | 996×996 [V] | none [V] | 6 [V] | 5 | yes, 3 z-planes (C06 Z01–Z03) [V] | 9.93 MB (1,986,560 B/file) [V] |
| source_3 | `r01c01f01p01-ch1sk1fk1fl1.tiff` (Phenix-style) | 1080×1080 [V] | LZW [V] | 9 [V] | 5 | batch-dependent: CP_35 has 3 planes (ch6–8), CP59 has none [V] | **11.85 and 12.26 MB**, means over 2 full plates of 3,456 sites each; per-site range 8.7–14.2 MB [V] |
| source_6 | Yokogawa-style | 970×970 [V] | none [V] | 9 [V] | 5 | yes, 3 planes [V] | 9.42 MB [V] |
| source_8 | `HTS_A01_s1_w1<GUID>.tif`, plus `_thumb` files to skip | 1024×1024 [V] | none [V] | 9 [V] | 5 | none [V] | 10.53 MB [V] |
| source_10 | Yokogawa-style; plate dirs also hold `DC_DCAM*.tif` 2000×2000 camera-calibration files to skip | 1000×1000 [V] | none [V] | 6 [V] | 5 | yes, 3 planes [V] | 10.01 MB [V] |
| source_7 | Yokogawa-style | 1280×1080 [V] | none [V] | 9 (sites 0–8) [V] | 5 | none [V] | 13.84 MB [V] |
| source_15 | Phenix-style | 1080×1080 [V] | LZW [V] | 9 [V] | **4: no RNA channel in any of 6 batches** [V] | 3 planes [V] | about 10 MB [E] |
| source_1, 9, 11 | Phenix-style | 1080×1080 | LZW | 4, 4, 9 | 5 | varies | 12–15 MB [E, from 2 wells each] |
| source_5 | Yokogawa-style | not checked | none | 9 | 5 | 3 planes | 9.97 MB [V, 2 wells] |

**Channel order varies between sources.** Mito is `C02` in source_7, `C05` in source_10, `ch1` in source_3 and `ch4` in source_15 [V, from the `load_data.csv` headers]. The loader must therefore map channels to stains through the `URL_Orig*` columns of `load_data.csv`, never by filename. A unit test with fixture rows from four sources enforces this.

**Brightfield is dropped.** It exists in sources 2, 6, 10 and 5 and in some source_3 batches, but not in sources 8 or 7, nor in source_3 batch CP59 [V]. A 5-channel model that works in every source therefore uses only the five fluorescence stains, in the canonical order `[DNA, ER, RNA, AGP, Mito]`. A brightfield-aware variant is out of scope.

**source_15 is excluded from image models.** It has no RNA channel, which matches its missing 957 CellProfiler features in ShapeShift (`step2/jump_notes.md`). CellProfiler numbers can still be reported for it.

**Illumination correction.** JUMP provides one 2-D `.npy` illumination function per plate and channel, referenced in `load_data_with_illum.csv` as `URL_Illum{DNA,...}` [V]. Correction follows CellProfiler: `corrected = raw / illum`.
- **File sizes** [V] imply float32 at native image size for most sources: source_2 3,968,192 B = 996²×4 + 128; source_6 970²×4 + 128; source_8 1024²×4 + 128; source_3 1080²×4 + 128; source_10 1000²×4 + 128. source_7's 2,764,928 B equals 1280×1080×2 + 128, which suggests float16 [D].
- **Rule:** read the dtype and shape from the `.npy` header and never assume them. The pilot checks this for every source.
- **Where the files are:** `illum/` folders were confirmed for sources 2, 3, 6, 7, 8, 9, 10 and 13. For Phenix-style sources (1, 4, 11, 15) the illum folder is named by plate barcode, not by the measurement directory (confirmed for source_3: `illum/BR5867a3/`), so an empty listing under the measurement-directory name is probably a naming mismatch, not proof that the files are missing [E for 1, 4, 11, 15]. source_5 (Yokogawa-style) also gave an empty listing under its plate directory; that is unexplained and is checked in G0.
- **How the functions were computed** (plate-level smoothed averages, the standard JUMP CellProfiler illum pipeline) was not re-checked here [E].

**Pixel size** is known only for source_15: 0.598 µm/px at 20×, NA 0.4, 2×2 binning [V, `Metadata_ImageResolutionX` in its `load_data.csv`]. The others are unknown [E] and are tabulated in pilot task G0 (risk R7).

### 1.2 Subset

**Anchor set (ShapeShift).** ShapeShift's T3 working set is 175 compounds plus DMSO: 306 plates and 14,063 wells at 10 µM across 10 sources, plus 110 plates and 5,865 wells at 0.625 µM in source_7 [V, `step2/morph_wells_{high,low}.parquet`].

Compound wells per training/held-out source [V, computed from those parquets]:

| Source | ShapeShift compounds | Compound wells | Plates hosting them | Compounds with ≥2 wells on different plates | Compounds with ≥2 distinct well positions |
|---|---|---|---|---|---|
| source_2 | 174 | 316 | 64 | 96 | 55 |
| source_3 | 172 | 419 | 59 | 164 | 42 |
| source_8 | 170 | 240 | 53 | 54 | 56 |
| source_10 | 37 | 41 | 30 | 4 | 4 |
| **source_6 (held out)** | **171** | **300** | **48** | **91** | **49** |
| source_7 (0.625 µM) | 175 | 1,290 | 110 | – | 66 |

165 compounds appear in all of sources 2, 3, 6 and 8. 171 of source_6's compounds also appear in at least one training source [V].

**Why these sources.**
- Sources 2, 3 and 8 each hold about 170 of the anchor compounds. source_10 adds a fourth instrument and site layout cheaply.
- **source_6 is the held-out source** because it holds 171 anchor compounds and was also left out by DeepCellMorpho, the strongest released baseline (§2), so the held-out comparison is clean for both models.
- source_7 is not used for training because its dose confound (0.625 µM against 10 µM everywhere else) is exactly the ShapeShift lesson. It is evaluated separately as "unseen source plus different dose".

**Training pool (full plan).**
- **Plates:** every plate in sources 2, 3, 8 and 10 that hosts an anchor compound, 206 plates in total [V]. 20% of plates per source (42 plates) are held out, stratified by batch, to give a plate split.
- **Wells:** all wells on the 164 training plates, except two masked groups:
  - the anchor-compound wells, so that evaluation queries are never seen as images;
  - the 16 evaluation DMSO wells per plate (2,624 wells), which serve as the per-plate z-score reference at evaluation, so that training plates and source_6 are treated alike. The dev cache's 4 DMSO wells per plate are a subset of these.

  That leaves **59,522 wells** [D]. Target-2 and all-DMSO plates are also excluded from training. `test_splits.py` asserts that no masked well appears in any training shard.
- **Sites:** 3 random sites per well, fixed by seed, for 178,566 sites. At the measured bytes per site that is **1.91 TB** [D]: source_2 0.55, source_3 0.61, source_8 0.48, source_10 0.26 TB.
- **Tiles:** 2 tiles of 448×448 per site (centre-jittered, kept only if Otsu DNA foreground is at least 1%, as Kim et al. do) gives **357,132 tiles** per epoch [D]; the cell filter may drop a few percent [E].

**Evaluation pool (full plan)**, totalling 113,295 sites and 1.29 TB [D]:

| Set | Content | Sites | TB |
|---|---|---|---|
| E1 HIGH | All anchor-compound wells, all sites, plus 16 DMSO wells per plate, all sites, in sources 2, 3, 8, 10 and 6 | 42,837 | 0.45 [D] |
| E2 LOW | source_7 anchor wells plus 16 DMSO per plate | 27,450 | 0.38 [D] |
| E3 Target-2 | 4 Target-2 plates in each of sources 2, 3, 6, 8 and 10 (141 exist in cpg0016 [V, `metadata/plate.csv.gz`]), 4 sites per well | 30,720 | 0.32 [D] |
| E4 DMSO plates | 4 all-DMSO plates in each of source_3 (38 exist) and source_10 (8 exist) [V], 4 sites per well; used for the position probe without compound confound | 12,288 | 0.14 [D] |

- **Total streamed once: about 3.2 TB** (1.91 TB training plus 1.29 TB evaluation) [D].
- **Target-2 caveat:** every Target-2 plate uses a single platemap (`JUMP-Target-2_compound_platemap.tsv`, 384 rows [V](https://github.com/jump-cellpainting/JUMP-Target)). Within-source replicate mAP on Target-2 is therefore fully position-confounded. E3 is used only for cross-source and target-annotation retrieval.

**Cut-down subset.**
- Training: sources 2, 3 and 8 only, 2 sites per well, with the same masking, giving 50,723 wells, 101,446 sites (1.10 TB) and 202,892 tiles [D].
- Evaluation: E1 without source_10 (39,711 sites, 0.42 TB), E2 (0.38 TB) and 2 source_3 DMSO plates (0.04 TB), for 70,233 sites and 0.84 TB [D].
- Target-2 is dropped.

### 1.3 Streaming plan

Two kinds of job read from S3. Both read each object once.

1. **`build_cache` (training images), one pass.**
   - Download: a thread pool of 64 workers [E] issues anonymous GETs for the 5 fluorescence TIFFs of each sampled site, in parallel across plates.
   - Decode with `tifffile` plus `imagecodecs`; LZW only for Phenix sources.
   - Divide by the plate's illumination function.
   - Scale each channel per plate, using the 0.1st and 99.9th percentiles of illumination-corrected DMSO sites on that plate, sampled from 64 sites. This keeps differences in intensity between wells, which is biology; per-image min-max would remove them.
   - Clip, quantise to uint8, cut 2 tiles of 448², and run the cell-content filter.
   - Write WebDataset tar shards of about 1 GB to the pod's disk. Each entry is a zstd-compressed `uint8[5,448,448]` plus JSON metadata (source, batch, plate, well, site, compound, is_dmso, row, col).
   - Size: 358 GB raw (357,132 × 1.0 MB) [D]. zstd may bring it to 180–250 GB [E, ratio 0.5–0.7 assumed], but **storage is budgeted at the raw size**, in case it barely compresses.
   - A small dev cache (anchor wells of sources 2, 3 and 8, plus 4 DMSO wells per plate; about 13,400 sites, 5 channels × about 1000², uint8; about 70 GB raw [E]) is built in the same pass. It is used for evaluation during training.
2. **`embed` (evaluation), one pass per group of models.**
   - Sites stream from S3 and are **decoded once**, as raw uint16 per channel.
   - Each model then gets its **own preprocessing adapter**: ours and the ImageNet/random controls use the per-plate DMSO-percentile scaling above; DeepCellMorpho uses its per-image percentile clip plus min-max; OpenPhenom uses its own normaliser.
   - Whether illumination correction is applied, and the channel order, are set per baseline by the fidelity gate G-B (§2).
   - Tiles are cut per model (§4.6). Only site-level embeddings are written (parquet, float16).
   - Two passes are planned: baselines before any training, and our final checkpoints at the end.

**Deviation from "keep only embeddings and checkpoints", stated deliberately.** Re-streaming 2 TB every epoch (200 TB over 100 epochs) is not viable: over the internet it would take days per epoch, and it would put needless load on a sponsored bucket. The tile cache and dev cache are therefore stored on the pod for the life of the project. They are deterministic given `manifest.parquet`, the code version and the seed, they are budgeted (about $53 of storage at raw size, §5), and they are **deleted at the end**. Only embeddings, checkpoints, logs and metric tables are kept.

### 1.4 Egress and where to train

**Who pays for transfer.**
- Bucket access is anonymous: no AWS account and `--no-sign-request` [V](https://registry.opendata.aws/cellpainting-gallery/), and every anonymous listing and GET in this doc succeeded.
- Anonymous, unsigned requests cannot be billed to the requester (a Requester Pays bucket rejects them), so **our transfer cost is $0 whether we read from EC2 or from RunPod/Lambda**. That is the verified fact.
- The AWS Open Data FAQ says sponsored datasets have "distribution costs such as egress and cross-region data fees" covered, "including access by external users" [V](https://aws.amazon.com/opendata/faqs/). The registry page itself does not state that this bucket is in the sponsorship programme [V], so who ultimately pays is not our concern, but it is not confirmed either.

**Throughput.**
- **EC2 in us-east-1:** S3 to EC2 in the same region is fast. Instance network ranges from "up to 20 Gbps" on g6e.2xlarge to 100 Gbps on p5.4xlarge [V, Vantage]. Pulling 3.2 TB takes about 1 h at 1 GB/s [E].
- **NAT gateway trap:** an instance in a private subnet behind a NAT gateway pays NAT data processing of $0.045/GB (the Ohio rate on the AWS page; us-east-1 not shown [V](https://aws.amazon.com/vpc/pricing/)). That would be about **$144 for 3.2 TB** [D]. Use a public subnet or an **S3 gateway VPC endpoint**, which has "no data processing or hourly charges" [V, same page].
- **RunPod/Lambda:** transfer is still free to us. Throughput over the internet is unknown, assumed 100–300 MB/s [E], so the 2 TB cache build takes about 2–6 h. This is the reason for the one-pass cache.
- **Not comparable:** ShapeShift measured 30 MB/s on a home link (`step2/jump_notes.md` [V]).
- **Gate G0** (§5.4) measures throughput on a cheap pod before any GPU spend.

**Recommendation.** Train on **RunPod or Lambda**, where A100/H100 per-hour prices are lower (§5), streaming directly from the gallery into the pod's cache. Fall back to EC2 us-east-1 (p5.4xlarge spot, or g6e.2xlarge), with an S3 gateway endpoint, only if G0 measures less than 100 MB/s sustained from the pod.

---

## 2. Baselines

**Shared post-processing for every representation.** All baselines go through the same well-level aggregation and post-processing (§4.6). The CellProfiler baseline is **recomputed on exactly the evaluation wells**, with the same 16-DMSO-per-plate subset, robust z-scoring, whitening protocol and ε. ShapeShift's published figures are used only for continuity checks, never as the baseline. ShapeShift's all-DMSO variant is also reported, to show the 16-DMSO subset does not handicap CellProfiler.

| # | Baseline | What is downloadable | Licence | Training data, and leakage into our splits | Status |
|---|---|---|---|---|---|
| a | **CellProfiler** well-level profiles (4,762 features; source_15 has 3,805) | Per-plate parquet in `workspace/profiles/` [V, ShapeShift `jump_notes.md`] | CC0 (gallery) [V](https://registry.opendata.aws/cellpainting-gallery/) | No training. Feature selection and whitening are fit on DMSO. | **Ready.** ShapeShift's `jump_profiles.py` re-pulls the 6.1 GB of plates. |
| b1 | **DeepCellMorpho** (J&J; Abdelmoula et al., *Nature Methods* 2026, doi:10.1038/s41592-026-03270-z). **It exists.** | Git LFS weights in `models/weights/` [V](https://github.com/johnsonandjohnson/DeepCellMorpho): DINO ViT-B/16 (1.75 GB), DINOv2 ViT-B/16 (0.44 GB; its media URL answers HTTP 200 to a HEAD request), DINOv2 ViT-B/16 cross-batch, ViT-B/16 short-sequence, ViT-B/8, **DINOv2 ViT-S/16 (178 MB)**, MAE ViT-B/16 (1.35 GB) [V, LFS pointer sizes], and iTPN HiViT-B/16 (size not fetched). Also precomputed embeddings for JUMP Target-2 and all of source_6 (for example 740 MB for DINOv2 source_6), with train/test/control metadata pickles (240 MB, 1.06 GB, 0.77 GB) [V]. | Weights and embeddings: **CC BY-NC-SA 4.0**; code Apache-2.0 [V, README] | Training metadata lists sources 2, 3, 5, 10, 8, 7 and 4 (about 350k rows) [V, the output of their `help_read_emb_metadata.ipynb`]. **source_6 is unseen.** Our training-source evaluation wells were probably seen (transductive); this can be checked by downloading their 240 MB `jump_cp_train.pkl` in the project phase. | **Primary pretrained baseline.** Inputs are 480×480×5 for DINO/DINOv2 and 224 for MAE/iTPN; per-image percentile clipping plus min-max per channel [V, `models/dinos/util/datasets.py`]. Channel order is not documented [E]: confirm in the fidelity gate. |
| b2 | **Bayer DINO** (Kim et al., "Self-supervision advances morphological profiling…", *Sci Rep* 2025 [V](https://pmc.ncbi.nlm.nih.gov/articles/PMC11811211)) | The paper points to `github.com/Bayer-Group/CellPaintSSL`, which **returns 404** (2026-11-22) [V]. A third-party fork ([drippypale/cellpaintssl-fork](https://github.com/drippypale/cellpaintssl-fork), code BSD-3) links a Dropbox share of about 6 GB with "single-source" checkpoints. The share resolves to an **HTML folder page** (HTTP 200, text/html) [V, HEAD only]. Downloadability is **not verified**. | Weights CC BY-NC 4.0 per the fork README [V] | The paper's training set is sources 2, 3, **6** and 8 (564,272 images, ViT-S/16, 200 epochs, about 300 V100 GPU-hours) [V, paper]. Which source the Dropbox **"single-source" checkpoints** were trained on is **unknown**, so source_6 must be assumed seen. | Optional and secondary. If used, it is reported only on source_7 and E4, and flagged as possibly seen. |
| b3 | **Recursion OpenPhenom** (CA-MAE, ViT-S/16, channel-agnostic) | `model.safetensors` on Hugging Face [V](https://huggingface.co/recursionpharma/OpenPhenom), not gated, last modified 2026-03-04 [V, HF API] | **Recursion Non-Commercial EULA** ("similar to CC BY-NC-SA") [V, LICENSE] | RxRx3 plus JUMP-CP ORF and CRISPR plates; **no compound plates** [V, model card]. Clean on every split. | **Ready.** The card recommends PCA-CenterScale or TVN on controls; our pipeline already does the equivalent. Input crops are 256² [E, from the repo code, to confirm]. |
| b3' | Phenom-Beta | Cloud API through NVIDIA BioNeMo only; not weights [V](https://www.recursion.com/news/nothing-short-of-phenomenal-new-deep-learning-model-available-on-nvidias-bionemo-platform) | Non-commercial terms | – | Not used. |
| b4 | **JUMP gallery DL profiles**: CP-CNN (`cpcnn_zenodo_7114558`) and EfficientNetV2-S ImageNet-21k | Precomputed, in `cpg0016-jump/{source}/workspace_dl/{embeddings,profiles}/{model}/{batch}/{plate}/`. Per-well-site `embedding.parquet` for CP-CNN is 0.5–0.8 MB; per-plate profile parquets are 1.6 MB (CP-CNN) and 21 MB (EfficientNet) [V]. The CP-CNN profile has 384 rows and a 672-dim float32 `all_emb` column [V, one plate downloaded]. Both model folders exist under `workspace_dl/profiles/` for sources 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 and 13, with one batch folder per image batch (for example 13 for source_6 and 4 for source_8). source_4 also has `cpdistiller_mesmer_s41467_025_62193_z`. **source_15 has no `workspace_dl/`** [V]. **No DINO or other SSL embeddings were found in the gallery** [V, listings]. CP-CNN weights: Zenodo 7114558, `Cell_Painting_CNN_v1.hdf5`, 19 MB, **CC-BY-4.0** [V](https://zenodo.org/records/7114558). | CC0 (gallery) / CC-BY-4.0 | CP-CNN training data not re-checked [E] | **Ready, at no compute cost.** Aggregated from the per-site files with our own site→well rule. |
| b5 | **ImageNet DINO ViT-S/16, frozen**, with the patch embedding inflated to 5 channels. This is our own initialisation. | `dino_deitsmall16_pretrain.pth` [V](https://github.com/facebookresearch/dino) | Apache-2.0 [V] | Never saw cells | **Essential control:** it shows what our SSL training adds. |
| b6 | Random-init ViT-S/16, 5 channels | – | – | – | A floor; random convolutional features are often non-trivial on Cell Painting [E]. |

**Not checked:** other channel-adaptive models such as ChAda-ViT and CHAMMI models. No released CA-MAE weights other than OpenPhenom were found.

**"DeepCellMorpho" is verified,** contrary to the worry in the brief. The public GitHub repository was created 2026-06-22 and last pushed 2026-10-29 [V, GitHub API]. It has a *Nature Methods* paper and real LFS weights.

**Baseline fidelity gate (G-B, before any comparison is trusted).**
- **DeepCellMorpho:** our `embed` path, using their DINOv2 ViT-B/16 weights, must reproduce their shipped `without_post_processing/DINOv2_ViT_B16/jump_cp_source6.pkl` for about 20 wells, with cosine ≥ 0.98 at well level [E threshold]. This fixes channel order, crop grid and normalisation.
- **OpenPhenom:** reproduce the repo's `test_huggingface_mae.py` and sample-image outputs.
- **CP-CNN:** our site→well aggregate must correlate at r ≥ 0.99 with the gallery's per-plate `profiles/` vectors.
- If a gate fails, the baseline is reported with that failure flagged, never silently.

**Security.** DeepCellMorpho's embeddings and metadata are Python **pickles**, and `.pth` checkpoints are pickle-based. Load `.pth` files with `torch.load(..., weights_only=True)` where possible. Unpickle `.pkl` files only inside a throwaway container, convert them once to parquet, and never unpickle them in the main environment.

**Licence hygiene for the public repo.**
- Our code: Apache-2.0.
- Our trained weights: CC-BY-4.0. The data is CC0 and the initialisation is Apache-2.0.
- **Never redistribute** DeepCellMorpho or OpenPhenom weights or embeddings. Publish only metrics and figures derived from them, with citations, and state that those baselines are NC-licensed.

---

## 3. Own model

### 3.1 Choice: DINO ViT-S/16, initialised from ImageNet DINO, 5-channel input

**Why DINO rather than MAE.**
- In Kim et al., DINO beat CellProfiler, MAE and SimCLR [V, paper].
- DeepCellMorpho reports that "contrastive and hybrid SSL models show stronger robustness than masked-only approaches" [V, README].
- MAE embeddings also typically need heavy post-hoc correction (OpenPhenom's card says so [V]).

**Why ViT-S/16 rather than ResNet-50.**
- It is directly comparable with Kim et al. (ViT-S/16) and DeepCellMorpho's ViT-S/16 ablation.
- About 22M parameters [E] fit an A100 at batch 256 with multi-crop [E].
- DINO ViT-S/16 ImageNet weights are Apache-2.0.
- ResNet-50 is not included, even as an ablation, to save budget.

**Why initialise from ImageNet rather than from scratch.** It converges in fewer epochs. Our compute is about one third of Kim et al.'s:
- Kim: 113M samples [D] on about 300 V100-hours.
- Ours: 37M samples per seed [D].

From-scratch training is pre-registered ablation A3.

**5-channel input.** Inflate the ImageNet patch embedding `W ∈ R^{384×3×16×16}` to `W5 = repeat(mean_rgb(W), 5) × 3/5`, so that activations keep their scale. Brightfield is dropped (§1.1). Channel order is canonical, `[DNA, ER, RNA, AGP, Mito]`, mapped from `load_data.csv`.

### 3.2 DINO recipe (DINO defaults [V](https://github.com/facebookresearch/dino/blob/main/main_dino.py) unless noted)

| Setting | Value |
|---|---|
| Projection head | out_dim 65,536 [V default]; ablation-free to save budget |
| Teacher | EMA, momentum 0.996→1 (cosine) |
| Teacher temperature | 0.04→0.07, warm-up over 30 epochs: the README's "boost" setting [V], not the script default |
| Centering | momentum 0.9 |
| Freeze last layer | first epoch |
| Global crops | 2 × 224² taken from the 448² tile at **native resolution** with ±20% scale jitter, replacing DINO's RandomResizedCrop(0.4–1) [E choice: cell size is biology] |
| Local crops | 6 × 96² at native resolution (±20%); DINO's default is 8 [V], reduced for speed [E] |
| Optimiser | AdamW, lr 5e-4 × batch/256, 10 warm-up epochs then cosine to 1e-6; weight decay 0.04→0.4 (cosine) |
| Regularisation | gradient clip 3.0; drop-path 0.1 |
| Batch | 256 per step on one GPU |

### 3.3 Hand-written training loop (no Lightning)

- **Precision:** `torch.autocast("cuda", dtype=torch.bfloat16)` with fp32 master weights. Use no GradScaler with bf16; a `GradScaler` path is kept for fp16-only GPUs.
- **Fused kernels:** SDPA/flash attention through `F.scaled_dot_product_attention`. `torch.compile` is optional, behind a flag.
- **Gradient accumulation** (`accum_steps`, used only on 24–48 GB GPUs):
  - the loss is divided by `accum_steps`;
  - the teacher EMA, LR/WD schedule step and **center update happen once per optimiser step**, using batch-mean teacher outputs accumulated over the micro-batches;
  - the effective batch is logged.
- **Schedules:** per-iteration numpy arrays for lr, weight decay, teacher momentum and teacher temperature (DINO style), indexed by the global step, so that resume is exact.
- **EMA teacher:** `teacher = m·teacher + (1−m)·student` under `torch.no_grad()`, with parameters only.
- **Checkpoint/resume:**
  - Contents: student, teacher, both heads, optimiser, scaler, global step, epoch, DINO center, sampler state (shard order and offset), the RNG states of Python, numpy, torch and CUDA, the config and the git SHA.
  - Writing: atomically (write `.tmp`, then rename) every 30 min and at each epoch end. Keep the last 2 and every 25th epoch.
  - A SIGTERM handler checkpoints and exits, for spot or community-pod pre-emption.
  - `--resume auto` picks the newest valid checkpoint.
- **Config:** YAML is loaded into nested frozen `@dataclass`es (`DataCfg`, `ModelCfg`, `DinoCfg`, `OptimCfg`, `RunCfg`, `EvalCfg`), with `key.sub=value` CLI overrides. Unknown keys raise an error. The resolved config is saved in the run directory, and its hash names the run.
- **Seeds:** one `seed` drives Python, numpy and torch. Each worker is seeded with `seed + 1000·rank + worker_id + epoch·10⁴`. Shard order is a function of `(seed, epoch)`. GPU kernels are not bitwise-deterministic (cudnn.benchmark is on); a CPU test checks that resume is bitwise-identical.
- **Logging:**
  - Every 50 steps, to CSV and TensorBoard: loss, lr, wd, momentum, teacher temperature, grad norm, samples/s, data-wait fraction and GPU memory.
  - Collapse monitors: teacher-output entropy, KL(student‖teacher) and the per-dimension std of backbone features.
  - Every 10 epochs: dev metrics (§4).
  - Weights & Biases (free tier) is optional, behind `log.wandb=true`.

### 3.4 Data loader

- **Choice: WebDataset shards on local disk**, not per-sample S3 reads. Per-sample reads would need about 5 GETs and 10 MB per sample at 280 samples/s, about 2.8 GB/s, which is impossible over the internet [D].
- **Source pools:** one shard list per source. `RandomMix` draws **equal samples per source per batch**, 64 each from 2, 3, 8 and 10, so source_10 is oversampled about 2× [D]. DMSO tiles are capped at 20% of each batch. Each batch should span at least 16 plates [E].
- **Workers and buffers:** 8–10 workers [E], `pin_memory`, persistent workers, `prefetch_factor=4`, and a 5k-sample shuffle buffer.
- **CPU vs GPU work:** CPU does only zstd decode and the 448→crop index draw. **Augmentation runs on the GPU** in batch, because flips, rotations, intensity changes and blur are cheap tensor operations.
- **Bandwidth needed:** 280 samples/s × about 0.6 MB compressed ≈ 170 MB/s of disk reads, about 1–2 CPU cores [E]. The loader is benchmarked alone (`bench_io.py`) before the pilot.
- **Decoded-tile caching:** the shard cache *is* the decoded-tile cache, already illumination-corrected and quantised, so no second-level cache is needed. The OS page cache helps on 220+ GiB RAM hosts (Lambda) [V, spec].

### 3.5 Augmentations

**Used:**
- dihedral D4 (flips and 90° rotations);
- per-channel gain U(0.8, 1.25), offset N(0, 0.05) and gamma U(0.8, 1.25) with p = 0.5;
- Gaussian blur σ ∈ [0.1, 1.0] (p = 1.0 on global crop 1, 0.1 on crop 2, 0.5 on local crops, keeping DINO's asymmetry);
- Gaussian noise σ = 0.02 (p = 0.3);
- channel dropout, zeroing one random channel with p = 0.1. This makes the model robust to a missing stain and allows a later source_15 experiment.

**Not used:**
- RGB colour jitter, hue/saturation, greyscale or solarisation, because channels are distinct stains, not colours;
- large rescaling, because cell size carries information;
- elastic or affine shear, because it changes shape;
- mixing channels between images.

### 3.6 Batch-aware design

1. **Source-balanced batches** (§3.4); ablation A1 uses natural proportions.
2. **Per-plate intensity normalisation** from DMSO percentiles at cache time (§1.3); ablation A2 uses per-image percentiles, as DeepCellMorpho does.
3. **Post-hoc per-plate DMSO robust-z and DMSO-fitted whitening** at evaluation, identical for all methods (§4.6).
4. **Not in the main model:** adversarial source losses, and positive pairs across sources of the same compound. The latter would use compound labels, making the model weakly supervised; DeepCellMorpho's "cross-batch" variant may do something similar, but that was not verified. It is a possible follow-up, not part of this budget.

---

## 4. Evaluation

Metrics use **copairs** ([cytomining/copairs](https://github.com/cytomining/copairs)) mAP with its permutation p-values and BH correction. Every metric is computed for every representation on **identical wells** with **identical aggregation and post-processing**.

### 4.1 Splits and leakage

| Split | Wells | Ours | DeepCellMorpho | OpenPhenom | Bayer DINO | ImageNet / random / CellProfiler |
|---|---|---|---|---|---|---|
| S-src: held-out source | source_6, E1 | unseen | unseen [V] | unseen [V] | **seen** [V] | n/a |
| S-plate: held-out plates | anchor wells on the 42 held-out plates of sources 2, 3, 8, 10 (186 compound wells [D]) | unseen | probably seen | unseen | probably seen (except source_10) | n/a |
| S-cpd: held-out compounds on seen plates | anchor wells on training plates | images unseen, plates seen | probably seen | unseen | probably seen | n/a |
| S-dose: source_7 at 0.625 µM | E2 | unseen | **seen** [V] | unseen | unseen [V] | n/a |
| Target-2 | E3 | unseen | test set [E] | unseen | validation set [V] | n/a |

**source_6 is locked.** No retrieval, probe or other metric on source_6 is computed for any method until the end, when all methods are scored once, together. Embedding source_6 images is allowed: the baseline pass computes them, and the fidelity gate compares vectors on **non-anchor** source_6 wells only, without computing any metric. Dev decisions (ε, checkpoint choice, early stop) use only sources 2, 3 and 8.

### 4.2 Primary metric: cross-source replicate retrieval, other-compound negatives

- **Query:** each source_6 anchor-compound well (300 wells, 171 compounds).
- **Positives:** wells of the same compound in sources 2, 3, 8 and 10.
- **Negatives:** wells of other anchor compounds in those sources.
- **Score:** mAP per compound (mean AP over its query wells), averaged over compounds.

**Why this metric.**
- **Position-clean by construction.** ShapeShift found the positional effect is within-source: DMSO pairs at the same well ID have cosine 0.005 across sources against 0.070 within a source [V, `jump_notes.md`]. We **re-measure this per representation** (§4.4), because ShapeShift showed it only for CellProfiler. A stricter variant that also drops same-well-ID pairs across sources is reported.
- **Other-compound negatives avoid the saturated DMSO null.** ShapeShift's DMSO-based activity calls were 96% active [V].
- **Well powered.** Within-source different-position retrieval in source_6 has only 49 usable compounds, because 78 of 171 compounds have a single source_6 well [V, §1.2]. The cross-source query has all 171.
- **Tests batch robustness directly.**

### 4.3 Secondary metrics (Holm-corrected as a family)

| ID | Metric | Notes |
|---|---|---|
| M1 | Within-source replicate mAP, **different (source, well) positions and different plates only**, other-compound negatives | Pooled over source × compound units in sources 2, 3, 6 and 8 with ≥2 distinct positions (202 units [D]). Also reported with DMSO negatives, as continuous mAP only, no binary calls. |
| M2 | Same as M1 with **no position restriction** | Reported **only** next to M1, to show how much position inflates it (ShapeShift: replicate cosine 0.36 same position against 0.16 different, sphered [V]). |
| M3 | **MoA mAP** on compound consensus profiles | Median within source, then mean across sources 2, 3, 6 and 8. Tahoe `moa-fine` classes with ≥3 members (as in ShapeShift T3: 82 annotated units, 15 classes [V]). ShapeShift's CellProfiler value of 0.26 (null 0.11) [V] is a continuity check only. |
| M4 | Target-2 cross-source compound retrieval, and target-annotation mAP | E3. Within-source Target-2 replicate mAP is not reported, because the platemap is shared. |
| M5 | Cross-dose retrieval: source_7 queries (0.625 µM) against 10 µM references in the other sources | Exploratory, because of the dose confound. ShapeShift found source_7 activity is weak once position is controlled: 22.7% `active_strict` among compounds with different-position pairs [V]. |
| M6 | Plate and compound splits: cross-source retrieval (§4.2) run with training-source queries, reported separately for queries on held-out plates (S-plate) and on seen plates (S-cpd), with the gap between them | Diagnostic for plate memorisation. **Low power:** only 186 anchor wells sit on held-out plates [D]. |

### 4.4 Diagnostics (pre-registered as comparisons across methods, not pass/fail)

**Well-position probe.**
- A logistic regression (L2, C chosen by inner CV) predicts **row** (16 classes) and **column** (24) from well embeddings, with GroupKFold by plate.
- Fitted on (i) E4 all-DMSO plates, where there is no compound confound, and (ii) DMSO wells of E1.
- Reports balanced accuracy against chance (1/16 and 1/24), plus edge vs interior AUC.
- A lower score means less position leakage.

**Same-well similarity gap.** Mean cosine of DMSO pairs on different plates at the same well against different wells:
- within a source (ShapeShift CellProfiler: 0.068 vs 0.002 [V]);
- across sources at the same well ID (CellProfiler: 0.005 vs −0.001 [V]).
- This validates, per representation, the "different position" definition used in M1 and §4.2.

**Source and batch probe.**
- Predicts source (5 classes) from DMSO well embeddings, with GroupKFold by plate; and predicts batch within source.
- Reported before and after post-processing.
- Raw embeddings will probably separate source near ceiling for every method [E], so the comparison across methods is the result.
- Also reported: a silhouette score by source.

**Replicate retrieval restricted to different-position pairs** is M1. The gap M2 − M1 is reported per method as the "position inflation" of that method.

### 4.5 Statistics

- **Primary test:** a paired per-compound difference ΔAP = AP(ours) − AP(CellProfiler) on the 171 compounds, with a compound-level bootstrap (10,000 resamples) 95% CI and a one-sided Wilcoxon signed-rank test.
- **Seeds:** P1 must pass for **every seed whose run is fixed at G1**: 2 in the full plan, 1 in the reduced tiers. The cut-down plan has 2 seeds at ≥250/s and 1 otherwise.
- **Power:** if the per-compound ΔAP has SD of about 0.15 [E], the standard error is about 0.012 [D]. The test can then detect Δ of about 0.025.

### 4.6 Identical aggregation and post-processing (all image models)

1. **Tiles per site.** Use each model's native crop on the largest centred grid that fits:

   | Model | Grid | Coverage of a ~1000² site |
   |---|---|---|
   | Ours / ImageNet / random | 4×4 of 224² | 896² |
   | DeepCellMorpho DINO | 2×2 of 480² | 960² |
   | OpenPhenom | 3×3 of 256² | 768² |

   Sensitivity analysis: every model restricted to the common central 768² window.
2. **Embedding per tile:** CLS token (DeepCellMorpho MAE/iTPN follow their notebook: LayerNorm on CLS or pooled stages [V]).
3. **Site:** mean over tiles.
4. **Well:** mean over sites. CellProfiler is well-level already.
5. **Plate:** robust z against the plate's 16 DMSO wells, with a MAD floor and clipping at ±50.
6. **Whitening:** PCA-whitening fitted on DMSO wells of all evaluation plates, keeping 99% of variance. ε is chosen from {0.1, 1, 3, 10} **separately for each representation**, by the same rule on the dev set only (sources 2, 3, 8; M1-style different-position AUC), then frozen for that representation across all splits. A single shared ε would handicap whichever representation it was not tuned for. In ShapeShift, ε was tuned on all data, a mild leak that is not repeated here.
7. **Optional secondary:** Harmony on top of whitening. JUMP's batch-correction benchmark favoured such methods; the Arevalo et al. 2024 result was not re-checked here [E].

**CellProfiler** goes through steps 5–6 after ShapeShift's feature selection (NaN, constant, blocklist, outlier, variance and |r| > 0.9 filters), fit on dev-source DMSO.

### 4.7 ShapeShift hook

On the 175-compound set:
- **(i)** replicate reproducibility with ShapeShift's strict definition (different-position pairs against an other-compound null), comparing CellProfiler (74% at 10 µM [V]) with each embedding;
- **(ii)** M3 MoA mAP;
- **(iii) exploratory:** rerun ShapeShift's T3 morphology-vs-Tahoe Mantel test (`step2/compare_tiers.py`) with each embedding in place of CellProfiler.

There is no pre-registered claim for (iii). ShapeShift showed that the agreement ceiling is limited by Tahoe's noise, not morphology's (§"Verdict" in `STEP2_REPORT.md`).

---

## 5. Compute and cost

### 5.1 GPU prices (fetched 2026-11-22)

| Provider | GPU | On-demand $/h | Spot $/h | Source; is it current? |
|---|---|---|---|---|
| RunPod Secure / Community | A100 SXM 80 GB | 2.19 / 1.39 | not shown | [V](https://www.runpod.io/pricing); page says "Updated November 12, 2026", so **current** |
| RunPod | A100 PCIe 80 GB | 2.09 / 1.19 | – | same |
| RunPod | H100 PCIe 80 GB | 2.89 / 1.99 | – | same |
| RunPod | H100 SXM 80 GB | 4.39 / 2.69 | – | same |
| RunPod | L40S / L4 | 1.09 / 0.69 (Secure) | – | same |
| RunPod storage | network volume | $0.07/GB-mo under 1 TB; container disk $0.10/GB-mo | – | same |
| Lambda | 1× A100 40 GB (30 vCPU, 220 GiB RAM, 512 GiB SSD) | 1.99 | none | [V](https://lambda.com/pricing); **undated** page, "© 2026" |
| Lambda | 1× H100 PCIe (26 vCPU, 225 GiB, 1 TiB) | 3.79 | none | same |
| Lambda | 1× H100 SXM | 4.89 | none | same |
| AWS us-east-1 | p5.4xlarge (1× H100 80 GB, 16 vCPU, 3.8 TB NVMe, 100 Gbps) | 6.880 | 4.403 | [V](https://instances.vantage.sh/aws/ec2/p5.4xlarge), a **third-party aggregator, undated**; check the AWS console before use |
| AWS | g6e.2xlarge (1× L40S 48 GB, 8 vCPU, 450 GiB NVMe) | 2.242 | 2.056 | [V](https://instances.vantage.sh/aws/ec2/g6e.2xlarge), same caveat |
| AWS | g6.4xlarge (1× L4 24 GB, 16 vCPU, 600 GiB NVMe) | 1.323 | 0.943 | [V](https://instances.vantage.sh/aws/ec2/g6.4xlarge), same caveat |

RunPod's vCPU/RAM per A100 pod and its ingress terms are not listed [E]. Check them at pod creation; at least 8 vCPU are needed for the loader [E].

### 5.2 Throughput

**Anchors.**
1. The DINO README gives ViT-S/16 for 100 ImageNet epochs (1.281M images) on 8 GPUs in 1.75 days, with default multi-crop of 2×224 + 8×96 [V](https://github.com/facebookresearch/dino). That is **about 106 samples/s per GPU** [D]. The DINO paper used V100s; the README line itself does not name the GPU [E].
2. Kim et al. report about 300 V100 GPU-hours for DINO ViT-S on 564,272 images for 200 epochs [V]. That is **about 104 samples/s per V100** [D], which agrees with anchor 1.

**Planning rates (all [E]).** These assume A100 bf16 with SDPA is 2.5–3× a V100, and 6 local crops instead of 8:

| GPU | Samples/s |
|---|---|
| A100 | **280** |
| H100 PCIe | 420 |
| H100 SXM | 520 |
| L40S | 200 |
| L4 | 85 |

**Hours per main run** (100 epochs × 357,132 = 35.7M samples [D]), plus 10% for evaluation and checkpoints:

| Hardware | Hours [D from E] | Cost per seed |
|---|---|---|
| RunPod A100 SXM Secure | 39.0 | **$85** |
| RunPod A100 SXM Community | 39.0 | $54 |
| RunPod H100 PCIe | 26.0 | **$75** |
| Lambda A100 40 GB | 39.0 | $78 |
| AWS p5.4xlarge | 21.0 | $92 spot, $144 on-demand |
| AWS g6e.2xlarge L40S | 54.6 | $122 on-demand |
| AWS g6.4xlarge L4 | 128 | $170 on-demand |

**Choice.** Budget on RunPod Secure A100 SXM. The pilot benchmarks H100 PCIe for 30 min and switches if its $/sample is at least 10% lower. Community Cloud is acceptable for ablations, which tolerate pre-emption.

**Inference** [E]:
- ViT-S at 224 is data-bound.
- DeepCellMorpho ViT-B/16 at 480 runs at about 300 crops/s on an A100, or about 75 sites/s.
- An evaluation pass over 1.29 TB at 150 MB/s takes about 2.4 h, so plan **5 h** for the baselines pass and **4 h** for our final pass.

### 5.3 Budget

**Full plan, at the planning rate of 280 samples/s:**

| Item | GPU-hours | $ |
|---|---|---|
| G0: I/O benchmark on a cheap pod; eval harness and CellProfiler/CP-CNN on CPU (laptop) | ~1 | 2 |
| G-B: baseline inference pass (all frozen models, E1–E4) | 5 | 11 |
| Training-cache and dev-cache build (2 TB, overlapping with pilot setup) | 4 | 9 |
| Pilot: 2 h DINO, plus a 30 min H100 benchmark | 3 | 7 |
| Main run, seed 0 (100 epochs) | 39.0 | 85 |
| Main run, seed 1 (100 epochs) | 39.0 | 85 |
| Ablations A1 natural mix, A2 per-image normalisation, A3 from scratch (25 epochs each) | 29.2 | 64 |
| Final inference pass (2 seeds + 3 ablations, one decode) | 4 | 9 |
| Storage: 500 GB network volume (raw caches ≈ 430 GB + checkpoints) × ~1.5 months | – | 53 |
| Egress | – | 0 [V] |
| **Subtotal** | | **≈ 325** [D] |
| Contingency | | 25 |
| **Total** | | **≈ $350** |

**Cut-down plan:**

| Item | $ |
|---|---|
| Baseline inference pass (0.84 TB) | 7 |
| Pilot | 7 |
| Cache build (1.10 TB) | 6 |
| Main run, seed 0: 80 epochs × 202,892 tiles = 16.2M samples | 39 |
| Main run, seed 1 (only if G1 ≥ 250/s) | 39 |
| Ablation A1 at 25 epochs (only if G1 ≥ 250/s) | 12 |
| Final inference pass | 7 |
| Storage: 300 GB × 1 month (raw caches ≈ 275 GB) | 21 |
| **Total** | **≈ $138 at 280/s; ≈ $126 at 140/s with seed 1 and A1 dropped** [D from E] |

**Overall envelope.** $350 + $50 for the fusion project = $400, leaving $100 under the $500 cap.

### 5.4 Stop/go gates

Gates are ordered so that the cheapest checks come first and training money is spent last.

| Gate | When | Pass criteria | If it fails |
|---|---|---|---|
| **G0: data** ($0–2) | Before any GPU | From a RunPod CPU pod in the chosen region: ≥150 MB/s sustained S3 reads over 10 min. Decode, illumination and tiling at ≥40 sites/s on 8 vCPU. Channel map validated for 2, 3, 6, 7, 8, 10. Illum dtypes read. Pixel size tabulated per source. | Below 100 MB/s: move to EC2 us-east-1 with an S3 gateway endpoint (p5.4xlarge spot). Channel or illum surprises are fixed in code first. |
| **G-B: harness and baselines** (~$11) | Before training | Evaluation harness reproduces ShapeShift's CellProfiler replicate numbers within ±0.02 AUC on the same wells. Fidelity gates of §2 pass. CellProfiler, CP-CNN, ImageNet-frozen, random, DeepCellMorpho and OpenPhenom results exist on the **dev set** (source_6 stays locked). | Fix the harness. A baseline that cannot be reproduced is flagged. |
| **G1: pilot** (2 h, ~$7) | Before the main run | **(a) Throughput** ≥ **250 samples/s** on A100, about the break-even rate for the $350 plan (≈254/s with $91 of fixed costs) [D], with GPU busy ≥85% and data-wait <10%. **(b) No collapse:** teacher entropy is not at its minimum or maximum, and feature std > 0.1 × initial. **(c)** Dev cross-source mAP (§4.2 run on sources 2/3/8) ≥ ImageNet-frozen + 0.01. Beating CellProfiler after 2 h is **not** required, since that is unrealistic. | 200–250/s: seed 1 cut to 50 epochs, 2 ablations. 150–200/s: 1 seed + A1 only. Below 150/s: debug the loader or switch to H100, else run the cut-down plan. (b) or (c) failing: stop, inspect augmentations and normalisation, and re-pilot once ($7). |
| **G2: 25% of the main run** (~10 h, ~$22) | Epoch 25 | Dev cross-source mAP ≥ CellProfiler's dev value. Dev M1 (different-position) ≥ CellProfiler − 0.01. Position-probe accuracy not more than 1.5× CellProfiler's. | Stop the main run. Keep the checkpoint, skip seed 1 and the ablations (saving about $200), and write up an **evaluation study**: baselines against CellProfiler under position and batch controls. That is still a complete portfolio piece. |
| **G3: seed 0 done** | Before seed 1 and ablations | Dev metrics at epoch 100 ≥ epoch 50 (no late degradation). | Use the best dev checkpoint, chosen by the pre-registered dev rule (best dev cross-source mAP), and run ablations at 25 epochs only. |

---

## 6. Pre-registration, risks and success

`PREREG.md` is frozen and git-tagged `prereg-v1` before the G1 pilot. Any later change is labelled a deviation.

### 6.1 Pre-registered claims

| ID | Claim | Split | Comparison | Criterion |
|---|---|---|---|---|
| **P1 (primary)** | Ours > CellProfiler | Held-out source_6, §4.2 cross-source retrieval, other-compound negatives | Identical wells and post-processing | Every seed fixed at G1 has mean ΔAP > 0, with the compound-bootstrap 95% CI excluding 0 (one-sided Wilcoxon p < 0.05). |
| P2 | Ours > ImageNet-frozen DINO | same as P1 | – | same test; shows SSL on JUMP adds value |
| P3 | Ours ≥ CellProfiler on M1 (different-position, within-source) | sources 2, 3, 6, 8 | – | CI lower bound > −0.01 (non-inferiority) |
| P4 | Ours' position inflation (M2 − M1) ≤ CellProfiler's | sources 2, 3, 6, 8 | – | point estimate, with CI reported |

Secondaries P2–P4 and M3–M5 are Holm-corrected.

**Expectations stated in advance (not claims).**
- DeepCellMorpho DINOv2 ViT-B/16 will probably match or beat ours: it has more compute and a bigger model [E]. Getting within 0.02 of DeepCellMorpho ViT-S/16 on P1's metric would count as a strong result.
- OpenPhenom has never seen compound plates and will probably trail [E].
- Source probes will be high for every method [E].

### 6.2 Risks

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | S3 throughput to RunPod/Lambda is too low | Medium | G0; EC2 fallback with an S3 gateway endpoint. Never route through a NAT, which would add about $144. |
| R2 | Throughput below 280/s pushes costs up 2× | Medium | G1 break-even rule at 250/s with explicit plan reductions; H100 benchmark |
| R3 | SSL learns plate, source or position shortcuts (intensity, vignetting, edge wells) | High | Per-plate normalisation, illumination correction, source-balanced batches; probes and position inflation reported; P4 |
| R4 | Fails to beat CellProfiler cross-source: well-level whitened CellProfiler is strong, and we use about 1/3 of Kim's compute | Medium–high | G2 early stop; the fallback write-up is pre-planned |
| R5 | Low statistical power: 171 compounds, 78 with a single source_6 well | Medium | Cross-source primary; bootstrap CIs reported whatever the outcome; no over-reading of secondaries (ShapeShift lesson: 77 of 262 nominal hits were mostly noise [V]) |
| R6 | Baselines handicapped by preprocessing mismatch (channel order, normalisation) | Medium | Fidelity gate G-B against DeepCellMorpho's own source_6 embeddings |
| R7 | Pixel size and optics differ by source (996 to 1280 px fields), so resolution becomes source signal | Medium | Tabulate pixel size in G0; if spread > 15%, resample to a common µm/px at cache time (decided before the pilot) |
| R8 | Leakage: anchor wells or plates seen in training | Low for ours (masked) | Split table in §4.1; source_6 locked; leakage unit tests on the manifest |
| R9 | source_7 dose confound misread as batch robustness | – | Separate, exploratory M5 only |
| R10 | Non-commercial licences (DeepCellMorpho, OpenPhenom, Bayer) | Certain | Do not redistribute; metrics only; state this in the README |
| R11 | Untrusted pickles from DeepCellMorpho | Certain | Sandboxed conversion; `weights_only=True` |
| R12 | Price or availability changes; Secure-pod or spot pre-emption | Medium | Checkpoint every 30 min; SIGTERM handler; prices re-checked on launch day |
| R13 | The Bayer weights cannot be obtained (repo 404) | High | Optional baseline only |

### 6.3 Success

- **Minimum (portfolio-complete even if P1 fails):**
  - public repo with CI;
  - reproducible manifest, with source_6 locked;
  - evaluation of 6 baselines under position and batch controls;
  - honest negative or positive result on P1.
- **Target:** P1 and P2 pass; P3 non-inferior; position inflation ≤ CellProfiler's.
- **Stretch:** within 0.02 of DeepCellMorpho ViT-S/16 on P1's metric; and the ShapeShift hook shows higher strict replicate reproducibility than CellProfiler's 74%.

---

## 7. Repo layout (`cellpaint-encoder`)

```
cellpaint-encoder/
├── README.md                # results table, licences of baselines, how to reproduce
├── PREREG.md                # frozen copy of §6 (tag prereg-v1)
├── LICENSE                  # Apache-2.0 (code); weights CC-BY-4.0, noted in README
├── pyproject.toml           # pinned deps; extras: [train], [eval], [dev]
├── configs/
│   ├── data/{full,cutdown,pilot}.yaml
│   ├── train/{dino_vits16_full,dino_vits16_cutdown,pilot,abl_natural_mix,abl_perimage_norm,abl_scratch}.yaml
│   └── eval/{dev,final}.yaml
├── src/cellpaint_encoder/
│   ├── config.py            # dataclasses + YAML + CLI overrides, strict keys, config hash
│   ├── data/
│   │   ├── gallery.py       # anonymous S3 listing/GET, retries, byte counter
│   │   ├── load_data.py     # parse load_data(_with_illum).csv -> canonical [DNA,ER,RNA,AGP,Mito]
│   │   ├── manifest.py      # build split manifest (train/heldout plates, masked wells, E1–E4)
│   │   ├── tiff_io.py       # tifffile/imagecodecs decode, skip DC_DCAM/_thumb files
│   │   ├── illum.py         # npy header-checked load, divide
│   │   ├── normalize.py     # per-plate DMSO percentiles -> uint8
│   │   ├── tiles.py         # 448 tiles, Otsu DNA foreground filter
│   │   ├── shards.py        # WebDataset writer/reader (zstd uint8 + json)
│   │   ├── sampler.py       # source-balanced RandomMix, DMSO cap
│   │   └── augment_gpu.py   # D4, per-channel gain/offset/gamma, blur, noise, channel dropout, multi-crop
│   ├── models/
│   │   ├── vit.py           # ViT-S/16, 5-ch patch embed inflation, SDPA
│   │   ├── dino_head.py
│   │   └── dino_loss.py     # centering, temperatures
│   ├── train/
│   │   ├── loop.py          # hand-written loop: bf16 autocast, accumulation, clip, EMA, schedules
│   │   ├── schedules.py     # per-iteration lr/wd/momentum/temp arrays
│   │   ├── checkpoint.py    # atomic save, auto-resume, RNG + sampler state, SIGTERM
│   │   ├── logging.py       # CSV + TensorBoard (+ optional W&B), collapse monitors
│   │   └── seed.py
│   ├── embed/
│   │   ├── run.py           # one decode -> many encoders; site-level parquet
│   │   ├── grids.py         # per-model crop grids
│   │   └── baselines/{imagenet_dino.py, random_init.py, deepcellmorpho.py, openphenom.py, gallery_dl.py}
│   └── eval/
│       ├── aggregate.py     # tile->site->well
│       ├── profiles.py      # robust z, CellProfiler feature selection, DMSO-fit whitening (ported from ShapeShift MORPH_SPEC)
│       ├── retrieval.py     # copairs wrappers: cross-source, M1/M2 masks, MoA, Target-2
│       ├── probes.py        # position (row/col, edge), source/batch probes, same-well gap
│       ├── stats.py         # compound bootstrap, Wilcoxon, Holm
│       ├── shapeshift_hook.py
│       └── report.py        # tables + figures into results/
├── scripts/                 # thin CLIs: bench_io.py, build_manifest.py, build_cache.py, train.py, embed.py, evaluate.py
├── tests/
│   ├── fixtures/            # 4 load_data header rows (src 2/3/7/15), tiny npy illum, tiny tiffs (synthetic 32x32)
│   ├── test_config.py           # YAML->dataclass, override, unknown-key error
│   ├── test_load_data.py        # channel mapping per source; source_15 missing RNA raises
│   ├── test_tiff_illum.py       # decode synthetic tiff, illum divide, dtype from npy header
│   ├── test_tiles_normalize.py
│   ├── test_model_shapes.py     # 5-ch ViT-S tiny variant, patch-embed inflation preserves mean response
│   ├── test_dino_loss.py        # loss finite; centering update; temperature schedule
│   ├── test_train_smoke.py      # 3 optimizer steps on random [B,5,64,64], accum=2, CPU, loss decreases
│   ├── test_resume.py           # train 4 steps vs 2+resume+2: bitwise-identical params on CPU
│   ├── test_sampler.py          # per-source balance, DMSO cap
│   ├── test_aggregate.py
│   ├── test_retrieval.py        # synthetic embeddings with planted replicates: mAP ~1; shuffled ~null
│   ├── test_position_masks.py   # M1 excludes same (source,well); strict variant excludes same well ID
│   ├── test_probes.py           # planted position signal is detected; none -> chance
│   └── test_splits.py           # no anchor well in train shards; source_6 absent from train manifest
└── .github/workflows/ci.yml
```

**CI** (`.github/workflows/ci.yml`):
- **Platform:** `ubuntu-latest`, Python 3.11, CPU-only torch wheel (`--index-url https://download.pytorch.org/whl/cpu`), with pip cache.
- **Steps:** `ruff check`, `ruff format --check`, `pytest -q -m "not network and not gpu"`.
- **Test data:** tiny synthetic tensors (batch 4, 5×64×64, ViT with depth 2 and dim 64). No downloads; S3-touching tests are marked `network` and run only by hand.
- **Target time:** under 5 min.
- **Also run:** a `schedules`/`config` doc-test, and a check that every `configs/*.yaml` parses into the dataclasses.

---

## Appendix A: what was fetched to write this doc (2026-11-22)

**S3 listings** (anonymous `list-type=2`):
- `cpg0016-jump/`, which gives sources 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15;
- the image, illum and `load_data_csv` prefixes of sources 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15;
- **two complete source_3 plates**: 17,285 and 27,653 keys;
- `workspace_dl/` of source_2, and prefix checks for the other sources.

**Range reads:**
- TIFF headers (8 B plus 8 KB IFD) for one file in each of 12 sources;
- `load_data.csv` headers of ≤300 KB for sources 2, 3, 6, 7, 8, 9, 10, 13 and 15 (6 batches), and 4 KB for source_1.

**Whole small files:**
- one CP-CNN per-plate profile parquet (1.6 MB);
- `jump-cellpainting/datasets/metadata/plate.csv.gz` (9.7 KB);
- the JUMP-Target-2 platemap (385 lines).

**No image or weight file was downloaded.** DeepCellMorpho weight sizes come from LFS pointer files, plus a HEAD request on one media URL.

**Web pages:**
- DeepCellMorpho GitHub (README, tree, licence);
- Kim et al. on PMC (PMC11811211);
- the CellPaintSSL fork README;
- OpenPhenom on Hugging Face (README, LICENSE, API);
- the Recursion BioNeMo news item;
- Zenodo API record 7114558;
- the DINO README and `main_dino.py`;
- AWS Open Data registry entry and FAQ;
- AWS VPC pricing;
- RunPod and Lambda pricing;
- Vantage pages for p5.4xlarge, g6e.2xlarge and g6.4xlarge.

**ShapeShift inputs:**
- `STEP2_REPORT.md`, `step2/jump_notes.md`, `step2/MORPH_SPEC.md`;
- `step2/morph_wells_{high,low}.parquet`, metadata columns only.

## Appendix B: one listing per source behind the [V] rows of §1.1

Each link is an anonymous S3 listing (the first 20 keys of one plate) or a manifest whose header was read. Prefixes containing spaces are URL-encoded.

| Source | Plate image listing | `load_data.csv` (channel map) |
|---|---|---|
| source_2 | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_2/images/20210614_Batch_1/images/1053600674/) | [csv](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_2/workspace/load_data_csv/20210614_Batch_1/1053600674/load_data_with_illum.csv) |
| source_3 (CP59) | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_3/images/CP59/images/BR5867a3__2022-04-29T01_02_16-Measurement%201/) | [csv](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_3/workspace/load_data_csv/CP59/BR5867a3/load_data.csv) |
| source_3 (CP_35) | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_3/images/CP_35_all_Phenix1/images/BAY5870a__2021-11-19T20_37_57-Measurement1/) | [csv](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_3/workspace/load_data_csv/CP_35_all_Phenix1/BAY5870a/load_data.csv) |
| source_6 | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_6/images/p210824CPU2OS48hw384exp022JUMP/images/110000293081/) | [csv](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_6/workspace/load_data_csv/p210824CPU2OS48hw384exp022JUMP/110000293081/load_data.csv) |
| source_8 | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_8/images/J1/images/A1170383/) | [csv](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_8/workspace/load_data_csv/J1/A1170383/load_data.csv) |
| source_10 | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_10/images/2021_06_07_U2OS_48_hr_run3/images/Dest210607-135426/) | [csv](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_10/workspace/load_data_csv/2021_06_07_U2OS_48_hr_run3/Dest210607-135426/load_data.csv) |
| source_7 | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_7/images/20210723_Run2/images/CP2-SC1-13/) | [csv](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_7/workspace/load_data_csv/20210723_Run2/CP2-SC1-13/load_data.csv) |
| source_15 | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_15/images/2022_02_22_Batch1/images/PEC00001818__2022-02-22T12_44_57-Measurement3/) | [csv](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_15/workspace/load_data_csv/2022_02_22_Batch1/PEC00001818/load_data.csv) (no `URL_OrigRNA`) |
| source_1 | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_1/images/Batch3_20221010/images/UL001667__2022-10-11T14_00_28-Measurement1/) | [csv](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_1/workspace/load_data_csv/Batch3_20221010/UL001667/load_data.csv) |
| source_9 | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_9/images/20210914-Run9/images/GR00003300/) | – |
| source_11 | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_11/images/Batch3/images/EC000154__2021-09-26T02_45_04-Measurement1/) | – |
| source_5 | [listing](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_5/images/JUMPCPE-20210719-Run13_20210721_131445/images/P06_ADMJUM013/) | – |

**Illumination folders:** [source_6](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_6/images/p210824CPU2OS48hw384exp022JUMP/illum/110000293081/), [source_3](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_3/images/CP59/illum/BR5867a3/), [source_7](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_7/images/20210719_Run1/illum/CP1-SC1-01/).

**Gallery DL profiles:** [source_6](https://cellpainting-gallery.s3.amazonaws.com/?list-type=2&max-keys=20&prefix=cpg0016-jump/source_6/workspace_dl/profiles/&delimiter=/).

**Plate types:** [plate.csv.gz](https://github.com/jump-cellpainting/datasets/blob/main/metadata/plate.csv.gz), which gives 141 TARGET2 plates, 77 DMSO plates and 1,876 COMPOUND plates.
