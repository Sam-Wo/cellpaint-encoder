# PREREG: cellpaint-encoder

**Frozen 2026-11-24, before the G1 pilot and before any GPU run. Git tag `prereg-v1`.**

- **Design:** [DESIGN.md](DESIGN.md) as committed in the pre-registration commit `052d5ce37a5d2901c107582d7302e988fdc826d6` (2026-11-22).
- **Code and configs:** the analysis code at the tagged commit is part of this registration, in particular `src/cellpaint_encoder/eval/` and `configs/`.
- **Run hash:** `config_hash(configs/full.yaml)` = `config_hash(configs/pilot.yaml)` = **`c900494f6d`**.
- **Deviations:** any change after the tag is a deviation, logged in `DEVIATIONS.md` with its date, its reason, and whether any result (dev or held-out) had been seen when it was made.

**What had been seen at freeze time:**
- The CPU smoke run: 12 sites from 3 plates of sources 2 and 3. Its metrics are degenerate by construction.
- The TIFF headers and image sizes of one DNA image per batch for sources 2, 3, 6, 8 and 10, read for the pixel-size check. No metric was computed.
- The Target-2 annotation tables.
- ShapeShift's published CellProfiler numbers.

**What had not been computed:** no metric for any representation on any real evaluation set, dev or held out.

---

## 1. Claims (copied from DESIGN §6.1)

| ID | Claim | Split | Comparison | Criterion |
|---|---|---|---|---|
| **P1 (primary)** | Ours > CellProfiler | Held-out source_6, §2 cross-source retrieval, other-compound negatives | Identical wells and post-processing | Every seed fixed at G1 has mean ΔAP > 0, with the compound-bootstrap 95% CI excluding 0 (one-sided Wilcoxon p < 0.05). |
| P2 | Ours > ImageNet-frozen DINO | same as P1 | – | same test |
| P3 | Ours ≥ CellProfiler on M1 (different-position, within-source) | sources 2, 3, 6, 8 | – | CI lower bound > −0.01 (non-inferiority) |
| P4 | Ours' position inflation (M2 − M1) ≤ CellProfiler's | sources 2, 3, 6, 8 | – | point estimate, CI reported |

**Seeds.** P1 must pass for every seed whose run is fixed at G1:
- G1 throughput ≥ 250 samples/s: 2 seeds at 100 epochs.
- 200–250/s: seed 1 at 50 epochs.
- 150–200/s: 1 seed.
- Below 150/s: per DESIGN §5.4.

**Holm family (secondaries).**
- **Members:** P2, P3, M3, **M4a**, **M4b** and M5, Holm-corrected together at α = 0.05.
- **M4 counts as two members.** DESIGN §4.3 lists two M4 metrics, and the design gives no rule for combining them.
- **P4** has no p-value. It is a point estimate with a CI and is not a family member.
- **Absent members:** a member whose comparison cannot be computed is dropped from the family, and the drop is reported.

**Expectations, stated in advance (not claims; DESIGN §6.1):**
- DeepCellMorpho DINOv2 ViT-B/16 will probably match or beat ours.
- OpenPhenom will probably trail.
- Source probes will be high for every method.

## 2. Primary metric and test (DESIGN §4.2, §4.5)

- **Query:** each source_6 anchor-compound well (300 wells, 171 compounds).
- **Candidates:** anchor wells of the reference sources 2, 3, 8 and 10, never the query's own source.
- **Positives / negatives:** wells of the same compound are positives; wells of other anchor compounds are negatives.
- **Score:** AP per query, mean per compound, then mean over compounds.
- **Strict variant:** additionally drops candidates at the query's well ID. It is reported next to the primary.
- **Test:** paired per-compound ΔAP = AP(ours) − AP(CellProfiler) on the 171 compounds.
  - Compound-level bootstrap, 10,000 resamples, 95% CI.
  - One-sided Wilcoxon signed-rank test.
- **source_6 lock:** every row of source_6 is dropped before any statistic unless `eval.unlock_heldout=true`. The lock is lifted **once**, at final scoring, for all methods together (RUNBOOK §3.1).

## 3. Secondary metrics (DESIGN §4.3) and diagnostics (§4.4)

M1, M2, M3, M5 and M6 are as defined in DESIGN §4.3, and the diagnostics as in §4.4, with the code rules in §4 below.

**M4a. Target-2 cross-source compound retrieval (E3).** Each source queries the others; within-source Target-2 retrieval is not reported, because Target-2 has a single platemap.

**M4b. Target-annotation mAP (E3).** This definition was added before the freeze, because DESIGN named the metric without defining it.
- **Annotation table:** `JUMP-Target-2_compound_metadata.tsv` from jump-cellpainting/JUMP-Target @ `40ce99d4` (MIT).
  - Rows with `pert_type` ≠ `trt` or with any `control_type` (positive controls and DMSO) are controls.
  - Rows are collapsed per full InChIKey; a key's target set is the union of the pipe-separated `target_list` of its rows.
- **Compound mapping:**
  - Each E3 compound (`Metadata_JCP2022`; DMSO `JCP2022_033924` excluded) is mapped to its InChIKey through `metadata/compound.csv.gz` (jump-cellpainting/datasets @ `016e865f`, BSD-3) and joined on the full InChIKey.
  - Only if a compound has no full-key match is the 14-character connectivity block used, and only when that block matches exactly one Target-2 key. The rule used is recorded.
  - On the real files at freeze time all 301 compounds matched on the full key: 255 annotated treatments and 46 positive controls.
- **Who takes part:** annotated compounds are mapped treatments with ≥ 1 target. Controls are neither queries nor candidates.
- **Profiles:** compound consensus profiles use the M3 aggregation (post-processed well profiles, median within source, mean across sources). The sources are every E3 source present: 2, 3, 8 and 10 while source_6 is locked; 2, 3, 6, 8 and 10 at final scoring.
- **Counted targets:** only targets held by ≥ 3 annotated compounds in the scored set count. The metadata predicts 162 targets and 148 queries.
- **Query, candidates, positives:**
  - A query is an annotated compound holding ≥ 1 counted target.
  - Its candidates are all other annotated compounds, ranked by cosine similarity.
  - Positives share ≥ 1 counted target with the query; every other candidate is a negative.
- **Score:** AP is the mean of precision@k over the ranks of the positives. The headline is the mean AP over queries; n_queries and n_targets are reported.
- **Null:**
  - The annotated compounds' target sets are permuted among them 100 times, with a fixed seed, so every representation gets the same permutations.
  - Reported: null mean, SD and 95th percentile, and p = (1 + #null ≥ observed) / 101.
  - Per-query permutation p-values and BH q-values follow M3.
- **Comparison with CellProfiler:** paired per-query ΔAP on the queries common to both, with a compound bootstrap 95% CI and a one-sided Wilcoxon test, as for M3.

## 4. Evaluation rules fixed in code (RUNBOOK §6)

- **M1:** the entire candidate pool is restricted to other plates **and** other well positions in the same source. **M2:** only the query itself is removed.
- **ε (whitening) choice:**
  - Chosen per representation from {0.1, 1, 3, 10}.
  - The criterion is the dev different-position replicate-vs-other-compound AUC, on dev sources 2, 3 and 8 only.
  - Ties go to the larger ε.
  - If that AUC is undefined, ε = 3 (the middle element `grid[len // 2]`).
- **Robust z:**
  - reference: the plate's 16 evaluation DMSO wells;
  - MAD floor: 0.01 × the median over plates of the scale, and at least 1e-12;
  - clip at ±50.
- **Whitening:** PCA whitening fitted on the evaluation DMSO wells of all evaluation plates, keeping 99% of the variance.
- **Aggregation:** tiles → site (mean) → well (mean), with each model's native crop grid (DESIGN §4.6). The sensitivity analysis uses the common central 768² window.
- **CellProfiler:** ShapeShift's feature selection, fit on dev-source DMSO, then the same robust z and whitening.
- **Scoping:**
  - M1–M3 and the ShapeShift hook use sources 2, 3, 6 and 8 (`eval.m1_sources`);
  - the dev cache uses sources 2, 3 and 8 (`data.dev_sources`);
  - the primary references and M6 include source_10.

## 5. Decisions made after DESIGN.md, before any training

All were made at $0 on the laptop, with no metric computed for any representation.

1. **Pixel size (risk R7): resampling is on.**
   - Per-source µm/px comes from jump-cellpainting/datasets `metadata/microscope_config.csv` @ `b3b4555d`. TIFF tags cannot be used.
   - source_8's 0.7032 µm/px was confirmed from its MetaSeries metadata; source_3's 0.598 matches source_15's verified value.
   - The spread over the used sources is 0.598–0.703 µm/px, about 17.6%, which is above 15%.
   - **Target:** 0.65 µm/px, with tolerance 0.02. Sources 2, 10 and 6 (held out) keep native pixels bit for bit. source_3 is scaled ×0.920 and source_8 ×1.082.
   - **Order:** illumination correction and the normalisation's affine map at native resolution, then torch antialiased bilinear resampling (`align_corners=False`), then clip, round, tile.
   - **Applies to:** every image model, including the DeepCellMorpho and OpenPhenom adapters (DESIGN §4.6, identical preprocessing). Details are in RUNBOOK §4.
2. **E3 and E4 are part of the full plan** (DESIGN §1.2).
   - E3: 4 Target-2 plates from each of sources 2, 3, 6, 8 and 10.
   - E4: 4 all-DMSO plates from each of sources 3 and 10.
   - Plates are drawn with the seeded `extra_eval_plates`, from the pinned `plate.csv.gz` and `well.csv.gz` (datasets @ `016e865f`).
3. **M4b and the Holm family:** as in §1 and §3.
4. **Gradient clipping:** DINO's per-parameter clipping at 3.0 (`optim.clip_mode=per_param`), as in `main_dino.py`.
5. **Dev cache:** 4 DMSO wells per plate, as in DESIGN §1.3.
   - In-training dev metrics z-score against those 4 wells.
   - G1(c) scores ImageNet-frozen on the same cache.
   - G2 scores CellProfiler on exactly the dev-cache wells (`--restrict-to-dev-cache`).
   - Final scoring (§2–§4) uses the 16 evaluation DMSO wells.
6. **Pilot:** the G1 pilot runs the first 2 h of seed 0 (same run hash). The main run resumes its checkpoint.
7. **Gates:** G0–G3 as in DESIGN §5.4. The checkpoint choice at G3 is the best dev cross-source mAP over the `ckpt-epoch*` files.
8. **Compute:**
   - **Platform:** vast.ai, one on-demand A100 SXM4 80 GB instance for every stage (RUNBOOK §1, Option V).
   - **G0 failure** (S3 < 100 MB/s): first another host, then AWS.
   - **H100 benchmark:** skipped unless G1 measures < 250 samples/s.
   - **Spending:** the cost checkpoints of RUNBOOK §3 are scaled by `dph / 2.19`. Reassess when spend exceeds a checkpoint by more than 20%. The hard cap for Part B is $500 in total.
   - None of this changes the science.
9. **Licence:** the code is MIT. DESIGN §2 planned Apache-2.0. This is not a science change.

## 6. Deviations

None at freeze time. The first deviation creates `DEVIATIONS.md`.
