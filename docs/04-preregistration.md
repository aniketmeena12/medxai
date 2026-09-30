# 04: Pre-registered analysis plan

**Status:** frozen 2026-09-17, before any data download or training. Changes after this date go in
the change log at the bottom, with a reason, and are reported in the paper as deviations.
Target: MIDL 2027 (est. deadline early Dec 2026). Scope: dermatology only.

## 1. Question and hypotheses

**Question:** when a skin lesion classifier is given patient metadata, does its image evidence become
better grounded on the lesion, or does it rely less on the image ("offloading")?

- **H-ground:** fusion models put *more* attribution inside the lesion than the image-only model.
- **H-offload:** fusion models put *less* attribution inside the lesion, and a metadata shortcut
  further erodes image grounding.
- Both outcomes are reported as findings. Accuracy is a secondary, sanity-check result.

## 2. Data (all splits fixed before training, grouped so no patient/lesion crosses folds)

| ID | Dataset | Task | Split | Metadata used | Ground truth used |
|---|---|---|---|---|---|
| A | PAD-UFES-20 | 6-class | 5-fold, grouped by `patient_id`, stratified | 21 clinical fields | none (counterfactual test) |
| B | HAM10000 | 7-class | 5-fold, grouped by `lesion_id`, stratified | age, sex, localization | lesion masks (all 10,015) |
| C | ISIC 2019 trap sets (Bissoto, ECCV-W 2022) | binary | published splits; **splits 1–3** of 10 per bias level | age_approx, sex, anatom_site_general (join verified: see change log) | model-inferred artifact labels; masks for the 9,083 HAM images |
| D | DDI (secondary) | binary | external test only | none usable → fusion models get all-MISSING tokens | Fitzpatrick groups |

A and B are separate experiments; B's models are never evaluated on C, and vice versa.

## 3. Models (identical backbone, input size, recipe, and tuning budget)

B0 image-only · B1 metadata-only (gradient boosting) · B3 concatenation · B4 FiLM · B5 MetaBlock ·
M1 cross-attention (img2meta). Backbone ConvNeXt-T (ImageNet), 224², AMP. Hyperparameters tuned
once per model on an inner validation split of fold 0, seed 0, with the **same number of trials** for
every model; then frozen. Seeds 0, 1, 2.

## 4. Attribution

- **Primary:** Grad-CAM on the last image-encoder stage, for every model (comparable across models).
- **Secondary:** RISE (gradient-free); M1 raw cross-attention maps.
- **Sanity gate:** model-parameter randomization test (Adebayo et al., 2018). A method whose maps
  stay similar after randomization (mean SSIM ≥ 0.5) is excluded and the exclusion reported.
- Maps are computed for the **predicted** class (primary) and the **true** class (sensitivity).

## 5. Primary endpoints (the only results that carry the paper's claims)

| ID | RQ | Endpoint | Data |
|---|---|---|---|
| **P1** | Grounding | **Energy inside lesion mask**: share of positive attribution inside the mask (higher = better grounded) | B |
| **P2** | Image shortcut | **Normalized area under the degradation curve**: test AUROC across bias levels {0, 0.3, 0.5, 0.7, 0.9, 1} | C |
| **P3** | Offloading | **Grounding loss under metadata shortcut**: change in P1-style energy (HAM subset of C) from metadata bias 0 → 0.9, at image bias 0 | C + injected metadata bias (§6) |
| **P4** | Counterfactual | **Map shift** = 1 − Spearman ρ between maps before/after changing one field, image fixed. Pre-defined *irrelevant* fields: `has_piped_water`, `has_sewage_system`. *Relevant* fields (reported separately): `bleed`, `grew`, `itch`, `changed`, `elevation` | A (and B: `sex`) |
| **P5** | Reliance | **Modality reliance**: BACC drop when metadata is shuffled across patients vs. when image is replaced by the dataset mean image | A, B |

**Secondary:** balanced accuracy, macro-F1, macro AUROC (T1); pointing game; Otsu-thresholded IoU;
attention mass in artifact regions (C); DDI AUROC by Fitzpatrick group; all endpoints under RISE.

## 6. Metadata trap sets (contribution N1)

- **Primary field: `sex`**, chosen because it is only weakly related to malignancy, so resampling to a
  target correlation keeps enough images. Resample training data so that the correlation between
  sex and label is **ρ_meta ∈ {0, 0.5, 0.9}**; the test set uses the **reversed** correlation.
- **Secondary:** a synthetic binary token with the same ρ levels (fully controlled, reported as a
  robustness check).
- **2-D grid:** ρ_image ∈ {0, 0.5, 0.9} × ρ_meta ∈ {0, 0.5, 0.9}, models B0, B3, M1, splits 1–3.
- If resampling leaves < 1,000 training images in any cell, that cell is dropped and reported.

## 7. Statistics

- **Unit:** out-of-fold predictions pooled per seed; metrics averaged over seeds.
- **CIs:** 95% **patient-cluster bootstrap** (10,000 resamples, `np.random.default_rng(0)`). For C,
  95% t-interval across the 3 published splits (one seed per split; see change log).
- **Primary comparisons:** each fusion model (B3, B4, B5, M1) vs B0 on P1–P3 and P5 (16 tests);
  M1 vs B3, B4, B5 on P4 (B0 has zero shift by construction; 3 tests). **Holm–Bonferroni across all
  19**, family-wise α = 0.05.
- **"Same accuracy" claim:** TOST equivalence on BACC with margin ±2 points.
- **Decision rule:** H-ground supported if P1 > B0 (corrected) and P3 is not significantly negative.
  H-offload supported if P1 < B0 or P3 is significantly negative. Otherwise "mixed", reported as such.
- No endpoint, field list, threshold, or model is added or dropped after seeing test results.

## 8. Compute budget and cut order (re-estimated 2026-09-17 from a timing run)

Timing run (RTX 3050 6 GB, M1, PAD-UFES-20 fold 0): ~10 s/epoch for 1,575 training images, ~60 s
startup per fold, peak GPU memory ~1.6 GB. Estimates assume early stopping near 25 epochs and
scale with training-set size. B1 (gradient boosting) costs seconds and is not counted.

| Block | Runs | Per run | GPU hours | Where |
|---|---|---|---|---|
| A: 5 image models × 5 folds × 3 seeds | 75 | ~5 min | ~7 | laptop |
| B: 5 image models × 5 folds × 3 seeds | 75 | ~25 min | ~31 | Kaggle |
| C image-bias curve: 3 models × 6 levels × 3 splits × 1 seed | 54 | ~35 min | ~32 | laptop + Kaggle |
| C 2-D grid (extra cells): 3 models × 6 cells × 3 splits × 1 seed | 54 | ~35 min | ~32 | laptop + Kaggle |
| **Total** | **258** | | **~100 h** | |

If over budget, cut in this order, and report it: grid splits 3 → 2 · B5/B4 from block C · seeds on
B 3 → 2 · frozen-backbone feature caching for block A. **Never cut:** B0, M1, P1–P5, patient grouping.

## 9. Schedule (9 weeks from 17 Sep 2026)

| Week | Exit criterion |
|---|---|
| 1 | All datasets on disk with SHA256 + `SOURCE.md`; metadata join coverage for C verified; timing run done |
| 2 | Loaders for A–D; splits written; tests pass |
| 3–4 | Blocks A and B trained; T1 filled |
| 5 | Attribution + sanity gate; P1, P4, P5 computed |
| 6–7 | Block C (curve + grid); P2, P3 computed |
| 8 | DDI + secondary analyses; figures frozen |
| 9 | Paper written; submitted to MIDL 2027 |

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-09-17 | Plan frozen | — |
| 2026-09-17 | Backbone weights pinned to `convnext_tiny.fb_in1k` (ImageNet-1k) | timm's default `convnext_tiny` tag now resolves to ImageNet-12k weights; pinned to match "ImageNet" in §3. No results seen. |
| 2026-09-17 | **Block C lesion-leak filter (primary):** val/test images whose `lesion_id` also appears in training are removed before any metadata resampling. **Secondary:** published splits unfiltered, image-bias curve only, models B0 and M1, splits 1–3 (36 runs, ~21 GPU h), reported descriptively for comparison with Bissoto et al. | The published splits are per image: ~2,500–3,000 of ~6,200 test images (≈48%) share a lesion with training. Filtered test sets keep 3,086–3,647 images; malignant rate drops from ~18% to 11–15%; test artifact rates shift by ≤ 8.4 points. Measured before any trap-set training. |
| 2026-09-17 | Tuning specified (§3 said only "same number of trials"): grid `lr` ∈ {1e-4, 3e-4, 1e-3} × `head_dropout` ∈ {0.1, 0.3} = 6 trials per model, per dataset A and B; fold 0, seed 0, selected on inner-validation balanced accuracy; test fold never loaded. Other hyperparameters fixed at `configs/base.yaml`. Block C reuses the HAM10000 settings. Adds ~15 GPU h. | Detail needed before implementation. Written before any tuning or training. |
| 2026-09-17 | Block C (trap sets) uses **one seed (0) per published split** instead of 3 seeds; CIs are t-intervals across the 3 splits | The original §8 run counts omitted seeds; with 3 seeds block C was 324 runs (~190 GPU h), infeasible. Splits already provide run-to-run variation. Decided before any trap-set training. |
| 2026-09-21 | **Block C extended to B4 and B5** (curve and grid), so the test family is the 19 tests §7 specifies. §7 counts 16 fusion-vs-B0 tests (4 models x P1/P2/P3/P5) but §6 and `configs/data_isic2019_trap.yaml` list only B0/B3/M1 for dataset C, which yields 15. §8 names "B5/B4 from block C" as a budget cut, and block C came in at ~7 GPU h against a ~64 h estimate, so no cut is needed. Decided after seeing P2/P3 for B0/B3/M1 but before any B4/B5 trap cell was run; the added tests are the ones §7 already specified, not new ones. | Removes an internal inconsistency in the plan rather than redefining the family after seeing results |
| 2026-09-19 | **Join coverage for C verified** (§2 marker resolved, no plan change): all **20,599** trap-set image ids have an ISIC 2019 metadata row; `sex` known for 98.2%, `age_approx` 97.9%, `anatom_site_general` 87.7%. **9,083** trap images are HAM10000 images (6,941 distinct HAM lesions), so they carry lesion masks — the evidence set for P3. Measured before any training on C. | Week-1 exit criterion; confirms the metadata trap field `sex` (§6) is available for ~all of C |
| 2026-09-19 | PAD-UFES-20 images re-downloaded from the Hugging Face mirror `SalmaneExploring/pad-ufes-20` (rev `086dd948`) instead of the Kaggle mirror; metadata still the official Mendeley file. HAM10000 images/masks and ISIC 2019 now come from their official records (Harvard Dataverse, ISIC S3) with published MD5s matched. See each `data/raw/*/SOURCE.md` | Project moved to a Linux machine with no Kaggle credentials; Mendeley exposes no direct image-folder link. Official sources were fast enough here. No results seen |
| 2026-09-17 | PAD-UFES-20 images from Kaggle mirror `mahdavi1202/skin-cancer`; metadata from official Mendeley file (SHA256 verified) | Mendeley download ~22 KB/s. Mirror has 2,298 images matching every official `img_id`; its CSV had identical values with different formatting. No results seen. |
