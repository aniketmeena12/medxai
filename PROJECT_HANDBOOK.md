# MedFusion-XAI — Project Handbook

> **One file to pick this project up on any machine.** It brings together the README, everything
> in `docs/`, the supervisor report (8 Sep 2026), the configs and the current state of the code.
> Last updated: **2026-09-17**. Owner: **Aniket Meena**.
>
> Where this file and a `docs/*.md` file disagree, the `docs/` file is the detailed source. Update
> both.

---

## ⚠️ Revision 2026-09-17: read before anything else

Research on 2026-09-17 (`docs/research/01–03`) changed the plan. **Decisions made:**

| Decision | Choice |
|---|---|
| Framing | **Grounding vs. offloading**: does patient metadata make the image evidence better grounded, or does the model stop looking? The architecture is *not* a contribution. |
| D-1 scope | **Dermatology only.** Chest X-ray moves to the journal extension. |
| D-2 venue | **MIDL 2027 full paper** (Validation Studies track if offered). Deadline estimated early Dec 2026 (2026 edition: 5 Dec 2025); watch https://2027.midl.io/. Fallback: MICCAI 2027 (~late Feb 2027), then iMIMIC / MultiTab workshops. |
| Analysis plan | Fixed in **`docs/04-preregistration.md`** before any training. |

**Corrections to facts elsewhere in this file** (sections below still contain the old text where marked ~~struck~~ or annotated):

| Old claim | Correct |
|---|---|
| Grad-CAM hit-rate 0.376 is the CheXlocalize number | Not in CheXlocalize. It is from MedicalPatchNet (arXiv 2509.07477). CheXlocalize Grad-CAM: mean hit rate ≈ 0.575, mIoU ≈ 0.282 |
| Hit rate uses a tolerance around the most-representative point | **Pointing game**: argmax pixel must be inside the GT mask, no tolerance. Maps thresholded with **Otsu** |
| NEATX reports drain AUC 0.940 / 0.770 | From **Oakden-Rayner et al., "Hidden Stratification", ACM CHIL 2020** (0.94 / 0.77) |
| Bissoto 6 bias levels in `debiasing-skin` on ISIC 2018 | Levels **{0, 0.3, 0.5, 0.7, 0.9, 1}** × **10 splits** in **`alceubissoto/artifact-generalization-skin`** (ISIC@ECCV 2022), on **ISIC 2019**. Binary malignant/benign. Artifact labels in `isic_inferred_wocarcinoma.csv` are **model-inferred probabilities** (thresholded at 0.6), not manual |
| Metadata fusion gain ≈ +2.5% on PAD-UFES-20 | That is accuracy from one handcrafted-feature paper. Deep-learning fusion typically gives **~7 BACC points**; published BACC baselines are 0.70–0.85. Include **MetaBlock** as a baseline |
| PAD-UFES-20 columns | Also has **`lesion_id`**. 2,298 images / 1,641 lesions / 1,373 patients |
| C1 cross-attention block is a contribution | Already published (MetaBlock 2021; Mridha & Islam, medRxiv 2026, same design on PAD-UFES-20) |

**New dataset facts (verified 2026-09-17):**
- **HAM10000** (Harvard Dataverse doi:10.7910/DVN/DBW86T, v4, **CC BY-NC 4.0**): 10,015 dermoscopic images (~2.8 GB) + `HAM10000_metadata` (age, sex, localization, lesion_id) + **`HAM10000_segmentations_lesion_tschandl.zip`** (10.8 MB, dermatologist-corrected binary lesion masks). Also on Kaggle.
- **9,083 of the 20,599** ISIC 2019 images in the Bissoto trap sets are HAM10000 images, so they have **lesion masks, metadata and artifact labels together**. This is the main evidence set for the paper.

---

## Contents

0. [Revision 2026-09-17](#️-revision-2026-09-17-read-before-anything-else)
1. [What the project is](#1-what-the-project-is)
2. [Current status (read this first)](#2-current-status-read-this-first)
3. [Setting up a new system](#3-setting-up-a-new-system)
4. [Repository layout](#4-repository-layout)
5. [Scope decision: what's in and what's cut](#5-scope-decision-whats-in-and-whats-cut)
6. [Datasets](#6-datasets)
7. [Architecture](#7-architecture)
8. [Experiment plan](#8-experiment-plan)
9. [Code reference: what exists](#9-code-reference-what-exists)
10. [Configs reference](#10-configs-reference)
11. [Work phases and milestones](#11-work-phases-and-milestones)
12. [Risks](#12-risks)
13. [Open decisions](#13-open-decisions)
14. [Rules and conventions](#14-rules-and-conventions)
15. [Known issues and TODOs in the code](#15-known-issues-and-todos-in-the-code)
16. [Reference library](#16-reference-library)
17. [Next actions checklist](#17-next-actions-checklist)

---

## 1. What the project is

> **Superseded 2026-09-17.** Current framing, contributions N1–N4 and research questions are in
> `docs/research/03-novelty-and-direction.md` and `docs/04-preregistration.md`. The text below is the
> original plan, kept for history.

**Current working title:** *Same Accuracy, Different Evidence: Does Patient Metadata Ground or
Distract Skin Lesion Classifiers?* (MIDL 2027)

**Original working title:** *Does cross-attention make the model look in the right place? Metadata-conditioned
image fusion and quantitative XAI evaluation in medical imaging.*

**Type:** Conference paper (originally MICCAI / ISBI tier, 8–9 pages; now MIDL 2027, up to 10 pages main / 14 validation track).

**Idea in one line:** Metadata-conditioned cross-attention doesn't just nudge accuracy. It changes
*where the model looks*. We measure that against human annotations and shortcut stress tests
instead of eyeballing heatmaps.

**Model:** image encoder + patient-metadata encoder, joined by **cross-attention**.

**What the paper does NOT claim:** a big accuracy gain. Prior work on PAD-UFES-20 reports only about
**+2.5%** from metadata fusion, which falls inside single-split noise on 2,298 images.

### Contributions

| # | Contribution | Evidence |
|---|---|---|
| C1 | Cross-attention fusion block that conditions image features on structured patient metadata | Architecture + ablation vs concat / FiLM / late fusion |
| C2 | Attention/Grad-CAM maps localize pathology closer to the human benchmark than image-only baselines | mIoU + hit-rate vs CheXlocalize radiologist segmentations; concept alignment vs SkinCon (and Derm7pt) |
| C3 | Fusion resists shortcut learning | Degradation curve across Bissoto's 6 bias levels; drain-stratified AUC via NEATX |
| C4 | Honest negative result: the accuracy gain is small | Reported with 95% CIs. This rules out "it just got better at everything" |

**C2 and C3 are the paper. C1 is how we get there. C4 makes the other results credible.**

### The four quantitative results the paper stands on

1. **Localization fidelity:** CheXlocalize hit-rate/mIoU compared with the published Grad-CAM number
   (hit-rate **0.376**). CheXlocalize found that Grad-CAM beat six other saliency methods, but all
   seven were significantly worse than radiologists, and the gap was largest on small,
   complex-shaped pathologies.
2. **Concept alignment:** do maps land on the 48 dermatologist-defined SkinCon concepts?
3. **Derm shortcut resistance:** performance vs bias level on the Bissoto trap sets.
4. **CXR shortcut resistance:** drain-present vs drain-absent pneumothorax AUC via NEATX
   (reference gap **0.940 / 0.770**).

### Why reviewers reject the naive version

- Grad-CAM figures with no quantitative evaluation.
- A headline accuracy delta from a single split, with no CI.
- No external test set.
- No shortcut analysis on datasets known for shortcuts (rulers, hair, ink, chest drains).
- Claiming cross-attention "explains" with no human-annotated reference.

The dataset plan exists to rule out all five.

---

## 2. Current status (read this first)

| Item | State |
|---|---|
| Stage | **Pre-registration frozen; all three datasets downloaded, verified (15/15) and processed; hyperparameter tuning under way.** Nothing from the full protocol (blocks A/B/C) has been run yet. |
| Current machine (2026-09-19) | **Linux**, RTX 5080 16 GB, project at `/home/ashok/Documents/medxai`, venv `.venv` (Python 3.12.13, torch 2.11.0+cu128). The system Python is 3.14, which torch does not ship wheels for — `uv venv --python 3.12 .venv` is how the venv was made. Note `ollama` runs on this box and reloads ~13 GB of GPU on demand; training fits alongside at ~1.6 GB but the margin is thin. |
| Previous machine (2026-09-17) | Windows 11, RTX 3050 6 GB, `C:\Projects\medfusion-xai`, venv `C:\medfusion-venv`. Source of the timing run in `docs/04-preregistration.md` §8 and of the B0/B3/B4 PAD-UFES-20 tuned configs. |
| Git | Branch `main`, 2 commits, no remote configured. |
| Data on disk | **29 GB.** PAD-UFES-20 (2,298 imgs) · HAM10000 (10,015 imgs + 10,015 masks) · ISIC 2019 (25,331 imgs) · both Bissoto repos at their pinned commits. Raw + `interim/` 256² caches + `processed/*/samples.csv`. All gitignored; provenance in each `data/raw/*/SOURCE.md`. |
| Data sources (2026-09-19) | Official where possible: HAM10000 images/masks from Harvard Dataverse (**published MD5s matched**), ISIC 2019 images from the ISIC S3, PAD-UFES-20 metadata from Mendeley (SHA256 matched). PAD-UFES-20 **images** come from the Hugging Face mirror `SalmaneExploring/pad-ufes-20` — Mendeley exposes no direct image-folder link and this machine has no Kaggle credentials. See `data/raw/pad-ufes-20/SOURCE.md` for the fidelity checks and the one gap left open. |
| Experiments run | Tuning only. PAD-UFES-20: B0/B3/B4 (prev. machine) + B5/M1 (here) → `configs/tuned/padufes20_*.yaml`. HAM10000 tuning queued. One debug run (`padufes20_M1_debug_001`, 1 epoch) as a smoke test; `complete_protocol: false`. |
| Tests | `pytest -q` → **42 passed** (2026-09-19, this machine, GPU present). |
| Supervisor report | `docs/MedFusion-XAI_Supervisor_Report.docx` (8 Sep 2026). It asks for decisions 1–4 in §13. |

### Pipeline as it actually runs now

```bash
bash scripts/download_data_official.sh          # official sources, no credentials
bash scripts/download_padufes_images.sh         # PAD images (HF mirror; TARGET=kaggle for Kaggle)
.venv/bin/python scripts/verify_data.py         # 15/15
.venv/bin/python scripts/prepare_data.py --dataset padufes20   # then ham10000, then isic2019
bash scripts/queue_tune.sh padufes20            # 6-trial grid per model
.venv/bin/python scripts/train.py --data configs/data_padufes20.yaml --model M1
```

`scripts/download_data.sh` (Kaggle mirrors) is kept as the alternative route.

### What's implemented vs scaffold

| Component | File | State |
|---|---|---|
| Metadata encoding (MISSING as a category), field dropout | `src/medfusion/data/metadata.py` | ✅ Implemented + tested |
| Grouped/stratified splits, leak assertions | `src/medfusion/data/splits.py` | ✅ Implemented + tested |
| Datasets, transforms, mask loading | `src/medfusion/data/dataset.py` | ✅ Implemented + tested |
| Trap sets (image traps + metadata resampling) | `src/medfusion/data/trap.py` | ✅ Implemented + tested |
| Image encoder (timm → spatial map, ConvNeXt + ViT) | `src/medfusion/models/encoder.py` | ✅ Implemented + tested |
| B0/B3/B4/B5/M1 + tokenizer; B1 tabular | `src/medfusion/models/` | ✅ Implemented + tested |
| Training + CV (folds × seeds, resumable, raw per-fold output) | `src/medfusion/train/engine.py` | ✅ Implemented; smoke-tested on real data |
| Trap-set cell runner | `src/medfusion/train/trap.py` | ✅ Implemented + tested |
| Grad-CAM, RISE, parameter-randomization sanity gate | `src/medfusion/xai/attribution.py` | ✅ Implemented + tested |
| Localization (energy-in-mask, pointing game, Otsu IoU) | `src/medfusion/eval/localization.py` | ✅ Implemented + tested |
| Shortcut (degradation area, offloading index, stratified AUC) | `src/medfusion/eval/shortcut.py` | ✅ Implemented + tested |
| Counterfactual map shift (P4) | `src/medfusion/eval/counterfactual.py` | ✅ Implemented + tested |
| Stats (bootstrap CIs, paired tests) | `src/medfusion/eval/stats.py` | ✅ Implemented + tested |
| Config loader, seeding, run dirs | `src/medfusion/utils/` | ✅ Implemented + tested |
| Download / verify / prepare / tune / train / train_trap | `scripts/` | ✅ Implemented; all run on real data except `train_trap.py` |
| **Endpoint driver** (compute P1–P5 from checkpoints → `results/`) | — | ❌ **Not written yet.** The library code exists; nothing wires it into tables/figures |
| Label mapping (needed only if DDI is added) | `docs/label_mapping.md` | ❌ Absent |
| DDI (dataset D, secondary) | — | ❌ Not obtained; needs the Stanford AIMI research-use agreement (manual) |

> Sections 3–10 below were written for the Windows machine and the older code layout (`prepare_padufes.py`,
> `explain.py`, `dataset_*.yaml`). Where they disagree with this section or with `README.md`, this
> section and the README are current.

---

## 3. Setting up a new system

### 3.1 Get the code there

The repo has **no commits and no remote**, so `git clone` won't work yet. Choose one:

**Option A: push to GitHub (recommended)**

```bash
# on the CURRENT machine
cd medfusion-xai
git add -A
git commit -m "Initial scaffold: docs, configs, fusion model, metrics, tests"
git branch -M main                       # the intended main branch is `main`
git remote add origin git@github.com:<you>/medfusion-xai.git   # create a PRIVATE repo first
git push -u origin main

# on the NEW machine
git clone git@github.com:<you>/medfusion-xai.git
cd medfusion-xai
```

`.gitignore` already excludes `data/*`, `.venv/`, checkpoints, `wandb/`, `mlruns/` and
`experiments/*/`. **Check `git status` before the first commit so that no data is included. The
DUAs prohibit redistribution.**

**Option B: copy the folder**, leaving out `.venv/` and `data/`:

```bash
rsync -av --exclude .venv --exclude 'data/raw/*/' --exclude 'data/interim/*/' \
      --exclude 'data/processed/*/' medfusion-xai/ user@newhost:~/medfusion-xai/
```

### 3.2 Python environment

Requires **Python ≥ 3.10** (developed on 3.12.3).

```bash
cd medfusion-xai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# GPU machine: install the CUDA build of torch FIRST, matching the driver's CUDA version.
# Pick the right index URL from https://pytorch.org/get-started/locally/ , e.g.:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU-only machine: skip this; the next line pulls CPU torch.

pip install -e ".[dev]"
```

**Dependencies** (from `pyproject.toml`): torch≥2.2, torchvision≥0.17, timm≥1.0, numpy≥1.26,
pandas≥2.2, scikit-learn≥1.4, pyyaml≥6.0, pillow≥10.2, tqdm≥4.66, matplotlib≥3.8,
opencv-python-headless≥4.9, grad-cam≥1.5, scipy≥1.12.
**Dev:** pytest≥8.0, ruff≥0.4, ipykernel≥6.29.

### 3.3 Verify

```bash
pytest -q                     # expect 13 passed
ruff check src scripts tests  # line-length 100
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi                    # on GPU machines
```

> On the original machine `nvidia-smi` failed with *"Driver/library version mismatch"* (NVML
> 580.173). That usually means the driver was updated without a reboot. Reboot before assuming there
> is no GPU.

### 3.4 Hardware to provision

| Component | Size |
|---|---|
| Raw data (both tracks) | ~77 GB |
| 224² preprocessed cache (~360k images) | ~17 GB |
| 384² cache (if localization needs it) | +35 GB |
| Checkpoints (~24 runs: 4 configs × 3 seeds × 2 tracks) | ~15 GB |
| Saliency / attention maps | ~2 GB |
| Extraction headroom (NIH tarballs, ISIC zip) | ~30 GB free |
| **Provision** | **~200 GB (a 250 GB volume)** + a GPU |

**Derm-only fallback:** ~35 GB raw + derived; provision 50 GB. Fits on a laptop.

### 3.5 Data on the new machine

Data is **never** in git. Download it again on every machine (§6.4), and for each dataset copy
`data/raw/SOURCE_TEMPLATE.md` → `data/raw/<dataset>/SOURCE.md` with the version, date, DOI/URL,
SHA256, DUA status and BibTeX. Once PAD-UFES-20 is unpacked:

```bash
# unzip into data/raw/pad-ufes-20/ so it contains images/ and metadata.csv
python scripts/prepare_padufes.py --raw data/raw/pad-ufes-20 --out data/processed/pad-ufes-20
# add --copy-images if symlinks are a problem (e.g. moving data between disks)
```

### 3.6 Intended pipeline (once the scaffolds are implemented)

```bash
python scripts/prepare_padufes.py --raw data/raw/pad-ufes-20 --out data/processed/pad-ufes-20
python scripts/train.py        --config configs/exp_derm_fusion.yaml
python scripts/explain.py      --run experiments/derm_fusion_001 --method gradcam
python scripts/evaluate_xai.py --run experiments/derm_fusion_001 --annotations skincon
```

`explain.py --method` accepts: `gradcam, gradcam_pp, ig, xrai, gradient, guided_backprop, cross_attention`.
`evaluate_xai.py --annotations` accepts: `chexlocalize, skincon, derm7pt, neatx, bissoto`.
`train.py --fold N` runs a single fold (**debug only**).

---

## 4. Repository layout

```
medfusion-xai/
├── PROJECT_HANDBOOK.md         ← this file
├── README.md                   Short overview + quickstart
├── pyproject.toml              Package "medfusion-xai" 0.1.0, src layout, ruff, pytest config
├── .gitignore                  data/*, runs, checkpoints, venv, wandb, mlruns
├── configs/                    YAML: one file per dataset, model, experiment (see §10)
│   ├── dataset_padufes20.yaml
│   ├── dataset_nih_cxr14.yaml
│   ├── model_baselines.yaml        B0–B4
│   ├── model_crossattn.yaml        M1 (ours)
│   ├── exp_derm_fusion.yaml        D1
│   ├── exp_derm_shortcut.yaml      D3
│   ├── exp_cxr_localization.yaml   X2
│   └── exp_cxr_shortcut.yaml       X3
├── data/                       ALL GITIGNORED except .gitkeep / SOURCE.md
│   ├── raw/                    Untouched downloads (+ SOURCE_TEMPLATE.md)
│   ├── interim/                Resized images, harmonized metadata CSVs
│   ├── processed/              Final splits, tokenized metadata, cached features
│   └── annotations/            CheXlocalize / NEATX / SkinCon / Bissoto label files
├── docs/
│   ├── 00-paper-brief.md       Claim, contributions, venues
│   ├── 01-datasets.md          Dataset dossier (DOIs, access routes)
│   ├── 02-access-checklist.md  Live download status tracker
│   ├── 03-experiment-plan.md   D1–D4, X1–X3, statistics, figure list
│   ├── 04-risks.md             R1–R9 with mitigations
│   ├── 05-open-decisions.md    D-1 … D-5
│   ├── 06-conference-scope.md  Scope cut, download list, references, phases
│   ├── label_mapping.md        PAD-UFES-20 ↔ external label spaces (TODO)
│   └── MedFusion-XAI_Supervisor_Report.docx
├── src/medfusion/
│   ├── data/                   Dataset classes, metadata tokenizers, split builders   (empty)
│   ├── models/fusion.py        MetadataTokenizer, CrossAttentionBlock, CrossAttentionFusion
│   ├── xai/                    Grad-CAM, attention rollout, map export                 (empty)
│   ├── eval/localization.py    iou, miou, hit_rate, pointing_game, attention_mass_in_region
│   ├── eval/shortcut.py        DegradationCurve, drain_stratified_auc
│   ├── eval/stats.py           mean_ci, paired_bootstrap
│   └── utils/                  Seeding, logging, config                                (empty)
├── scripts/                    prepare_padufes.py (done), train/explain/evaluate_xai (scaffolds)
├── experiments/                One subdir per run (gitignored); README.md has the rules
├── results/                    figures/ tables/ checkpoints/ (paper artifacts only)
├── notebooks/                  Exploration only; nothing in the paper comes from here
└── tests/                      test_fusion.py, test_eval.py
```

---

## 5. Scope decision: what's in and what's cut

Decided 2026-09-08. **A complete two-modality conference paper can be built with no PhysioNet
credentialing.** That takes about 800 GB of downloads and about 6 weeks of waiting off the critical
path, and all four results in §1 still stand.

### Dropped for now (kept for a journal extension; the same code interface reads them later)

| Dropped | Why it's safe to cut |
|---|---|
| MIMIC-CXR + MIMIC-IV | 558 GB + 2–6 weeks of credentialing, for metadata that can be partly replaced |
| VinDr-CXR | 192 GB; CheXlocalize already gives better (pixel-level) ground truth |
| PadChest | Separate multi-week application; a second generalization set is a luxury at 8 pages |
| BRAX | Third generalization set; redundant at conference length |
| Derm7pt | Second concept set; SkinCon's 48 concepts already support the concept claim |

### What the cut costs

Without MIMIC-IV, the CXR metadata is only **age, sex and view position** (3 fields), compared with
PAD-UFES-20's **21 usable fields**. **Dermatology carries the "metadata-conditioned" premise**;
CXR is supporting evidence that the mechanism transfers. Two points keep that defensible:

1. **View position (AP vs PA) is a documented shortcut.** Portable AP films correlate with sicker,
   supine patients, so conditioning on view and showing the model attends to pathology is a real
   experiment.
2. All four quantitative results still stand.

**Still recommended:** file the PhysioNet credentialing application anyway, in the background
(~1 hour; needs a supervisor reference). It unlocks the journal version and blocks nothing.

---

## 6. Datasets

Three dataset roles, each answering one reviewer objection:

| Role | Purpose | Derm | CXR |
|---|---|---|---|
| Metadata-rich training set | Fusion needs real per-patient features | PAD-UFES-20 | NIH-CXR14 (MIMIC later) |
| External generalization set | Survives domain / skin-tone shift | DDI, Fitzpatrick17k (Derm7pt later) | (PadChest, BRAX, VinDr later) |
| Annotated XAI ground truth | Makes the XAI claim measurable | SkinCon, Bissoto artifacts + trap sets | CheXlocalize, NEATX |

> All numbers come from the project brief. **Check them against the primary source before they go
> into the paper**, and convert every attribution into a proper citation.

### 6.1 Dermatology track (primary, ~19 GB)

| Dataset | Size | Role | Access |
|---|---|---|---|
| **PAD-UFES-20** | 3.5 GB | Fusion training set | Instant: Mendeley Data, DOI `10.17632/zr7vgbcyr2.1` |
| Fitzpatrick17k | 1.3 GB | SkinCon image substrate (3,230 imgs) + skin-tone shift | Instant |
| Fitzpatrick17k-C | 50 MB | Corrected labels / de-duplication | Instant |
| ⭐ **SkinCon** | 10 MB | Concept-alignment ground truth (48 concepts, 3,886 images) | Instant |
| DDI | 239 MB | SkinCon's other 656 images + skin-tone-balanced external test | Redivis, same day |
| Bissoto repos (×2) | 100 MB | Trap sets + artifact annotations (shortcut test) | Instant: `github.com/alceubissoto/debiasing-skin` |
| ISIC 2018 Task 1&2 training input | 12.61 GB | ⚠️ **Required** image substrate for the Bissoto trap sets | Instant |

**PAD-UFES-20 details**
- **2,298** smartphone **clinical** (not dermoscopic) images; **six** classes: BCC, SCC, ACK, SEK, MEL, NEV.
- CSV with **up to 26 clinical features per lesion** (21 usable): age, anatomical site, Fitzpatrick
  type, diameters, itch, bleed, family history, etc. Many values are missing.
- **58.4% biopsy-proven**, including **100% of the cancers**.
- **Downside:** small, so cross-attention overfits. Cross-validation is mandatory.
- Multiple lesions/images per patient → **split at the patient level only**.

**SkinCon:** dense annotations for **3,230** Fitzpatrick17k + **656** DDI images with **48 clinical
concepts** (**22** have ≥50 images each; use those for statistics).

**Bissoto:** manual annotations of **7 artifacts** (dark corners, hair, gel borders, gel bubbles,
rulers, ink markings, patches) on ISIC 2018 Task 1&2 (**2,594** images) and the Interactive Atlas
of Dermoscopy (**872** images). Includes **pre-split trap sets at 6 bias levels**, from random split
to artifact–label correlation **reversed** between train and test.

**Deferred derm datasets**
- **Derm7pt** (~1,011 cases): dermoscopic + clinical images + metadata + **12 concept labels** from
  the 7-point checklist. Request via the SFU MIA lab page (not instant).
- **ISIC 2019/2020**: thin metadata (age, sex, site); ISIC 2020 = **33,126** dermoscopic images from
  2,000+ patients. Useful for **pretraining the image branch** only.

### 6.2 Chest X-ray track (confirmatory, ~58 GB)

| Dataset | Size | Role | Access |
|---|---|---|---|
| NIH ChestX-ray14 | 45 GB | Training set + NEATX substrate. 112,120 frontal images, 30,805 patients, 14 labels, age/sex/view | Instant, open |
| CheXpert (downsampled) | 11 GB | Training set for the CheXlocalize evaluation | Redivis, same day |
| ⭐ **CheXlocalize** | ~2 GB | Radiologist **pixel segmentations for 10 pathologies** on CheXpert val/test, plus a benchmark set (segmentations + most-representative points) from 3 more board-certified radiologists | Redivis, same day |
| NEATX | ~100 MB | Chest-drain annotations: ~**3.5k** for CXR14, ~**1k** across **4 tube types** in PadChest | Instant, open |

**Why NEATX matters:** drains are inserted to *treat* pneumothorax, so a model can learn the
treatment instead of the disease and then fail on untreated patients.

**Deferred CXR datasets:** MIMIC-CXR-JPG + MIMIC-IV (richest metadata: labs, vitals, demographics,
notes, joined on `subject_id`; credentialed PhysioNet + DUA + CITI), PadChest (160,861 images,
67,625 patients, Spanish reports), VinDr-CXR / BRAX (bounding boxes; BRAX is Brazilian, which is
useful for domain shift).

### 6.3 Pipelines per track

```
Derm:  train PAD-UFES-20 (metadata fusion)
         -> external test on DDI (Derm7pt later)
         -> shortcut test on Bissoto trap sets
         -> concept alignment via SkinCon

CXR:   train NIH-CXR14 (age/sex/view as metadata tokens; MIMIC later via config swap)
         -> Grad-CAM vs CheXlocalize (mIoU, hit-rate)
         -> drain-stratified evaluation via NEATX
```

### 6.4 Download order (doable in one day)

1. **Instant, no account (~62 GB):** PAD-UFES-20 · Fitzpatrick17k + Fitzpatrick17k-C · SkinCon CSVs ·
   both Bissoto repos · ISIC 2018 Task 1&2 zip · NIH-CXR14 · NEATX.
2. **One Redivis account**; accept the Stanford research-use agreement 3 times (~13 GB): DDI ·
   CheXpert (downsampled) · CheXlocalize. Same day to +1 business day.
3. Nothing else blocks: no CITI course, no reference chase, no BIMCV form.

### 6.5 Access tracker (mirror of `docs/02-access-checklist.md`; all TODO)

| Tier | Datasets |
|---|---|
| Immediate, no application | PAD-UFES-20, NIH-CXR14, Bissoto, SkinCon, Fitzpatrick17k, ISIC 2019/2020 |
| Registration (days) | CheXpert, CheXlocalize, PadChest, Derm7pt, DDI, NEATX |
| Credentialed (2–6 weeks) | MIMIC-CXR-JPG, MIMIC-IV, VinDr-CXR, BRAX |

**PhysioNet critical path:** create account → complete CITI "Data or Specimens Only Research" and
upload the report → submit credentialing application (needs a supervisor reference) → sign the DUA
for each dataset.

### 6.6 Ethics / compliance

- All DUAs prohibit redistribution. `data/` stays gitignored. **Do not upload data anywhere.**
- No patient-level data in `results/`, figures or the paper beyond what each DUA permits.
- Record the exact version/release date for every download in `data/raw/<ds>/SOURCE.md`.

---

## 7. Architecture

```
image    --> image encoder (ConvNeXt-T or ViT-B/16; ImageNet or ISIC-pretrained) --> patch tokens P  (B, N, D)
metadata --> MetadataTokenizer: one token per clinical field -------------------> meta tokens  M  (B, M, D)

fusion (img2meta):   P' = P + CrossAttn(Q=P, K=M, V=M)  (+ FFN), n_blocks times
mirror (meta2img):   M' = M + CrossAttn(Q=M, K=P, V=P)
bidirectional:       both; pooled outputs concatenated

head:  mean-pool -> LayerNorm -> Linear -> logits
```

**Explanation channels, all compared:**
1. Grad-CAM on the image encoder's last block.
2. Grad-CAM on the fused representation.
3. Raw cross-attention weights (`CrossAttentionFusion.attention_maps`), giving a map of shape
   `(B, n_fields, H, W)`: for each metadata field, where in the image it was consulted. **Only
   `img2meta` produces spatial maps directly.**

**Regularization for 2,298 images:** heavy augmentation, layer-wise LR decay, frozen early stages,
dropout, **metadata field dropout** (random masking of fields; also serves as a robustness ablation),
ISIC-pretrained image branch, early stopping on the CV mean.

### Baselines (all required for table T1)

| ID | Model | Purpose |
|---|---|---|
| B0 | Image-only | The reference everything is compared with |
| B1 | Metadata-only (gradient boosting / MLP on the CSV) | Signal in metadata alone |
| B2 | Late fusion (logit averaging) | Cheapest fusion |
| B3 | Early concat (image emb ⊕ metadata emb) | The usual literature "fusion"; **the honest competitor** |
| B4 | FiLM conditioning | Strong non-attention conditioning |
| **M1** | **Cross-attention (ours)** | The proposal |

---

## 8. Experiment plan

**Rule:** every experiment produces a numbered table or figure in `results/`, or it isn't run.

### Derm track

**D1: Fusion training on PAD-UFES-20**
- 5-fold **patient-stratified** CV × 3 seeds. Never split a patient across folds.
- Report balanced accuracy, macro-F1, per-class AUC, **with 95% CI across folds × seeds**.
- Expect about +2.5% over B0. Report it plainly.

**D2: External generalization**
- D1 checkpoints evaluated **zero-shot** (no retraining) on DDI (and Derm7pt if obtained).
- Stratify DDI by **Fitzpatrick skin tone** (fairness).
- **`docs/label_mapping.md` must be filled in before any external number is produced.**

**D3: Shortcut stress test (Bissoto trap sets)**
- Train B0, B3, M1 at each of the **6 bias levels**; seeds 0, 1, 2.
- Deliverable: **degradation curve** (AUC vs bias level, one line per model). The curve's *shape* is
  the result.
- Secondary: **attention mass inside the 7 artifact regions** (lower is better).
- Under compute pressure, cut seeds before bias levels.

**D4: Concept alignment**
- SkinCon: overlap between saliency and each concept's presence (the 22 concepts with ≥50 images).
- Derm7pt (if obtained): align with the 12 seven-point-checklist concepts.
- Deliverable: concept-alignment score per model; M1 vs B0 vs B3.

### CXR track

**X1: Fusion training.** NIH-CXR14 first; metadata tokens = age, sex, view position. Patient-level
splits, multi-label BCE, per-pathology AUC.

**X2: Localization vs CheXlocalize**
- Saliency on CheXpert val/test with 7 methods: `gradcam, gradcam_pp, ig, xrai, gradient,
  guided_backprop` + `cross_attention`.
- Metrics: **mIoU**, **hit-rate** (including against the most-representative points), pointing game.
- **Replicate first** (Grad-CAM best of seven; all trail radiologists; largest gap on small, complex
  shapes). **Then extend:** does M1 close the gap?
- Stratify by pathology, lesion-area quartile, shape complexity.

**X3: Drain-stratified evaluation (NEATX)**
- Split pneumothorax test cases into drain-present / drain-absent.
- Deliverable: AUC per stratum and the **gap**. Secondary: attention mass inside drain regions.

### Statistics (non-negotiable)

- ≥3 seeds × 5 folds for every headline number; report **mean ± 95% CI**.
- Paired bootstrap or DeLong for AUC comparisons; **correct for multiple comparisons**.
- `metrics.json` stores per-fold/per-seed values; CIs are recomputed from raw values.
- **Never report a single-split ~2.5% accuracy delta as a result.**

### Paper figures and tables

| ID | Artifact | Source |
|---|---|---|
| F1 | Architecture diagram | none (drawn by hand) |
| F2 | Degradation curve across Bissoto bias levels | D3 |
| F3 | mIoU / hit-rate vs radiologist benchmark, stratified by lesion size | X2 |
| F4 | Drain-present vs drain-absent AUC gap | X3 |
| F5 | Qualitative B0 vs M1 saliency with segmentation overlay | X2 / D4 |
| T1 | Main results + ablation (B0–B4, M1) | D1 / X1 |
| T2 | External generalization by Fitzpatrick tone | D2 |
| T3 | Concept alignment scores | D4 |

---

## 9. Code reference: what exists

### `src/medfusion/models/fusion.py`

**`MetadataTokenizer(cat_cardinalities: list[int], n_continuous: int, dim: int)`**
- One `nn.Embedding(c + 1, dim)` per categorical field; **index `c` = MISSING**.
- One `nn.Linear(1, dim)` per continuous field, plus a learned `cont_missing` token used where
  `cont_mask` is False.
- Adds learned per-field positional embeddings, then LayerNorm.
- `forward(cat (B,n_cat) long, cont (B,n_cont) float, cont_mask (B,n_cont) bool) -> (B, n_cat+n_cont, dim)`.
- Missing values are treated as a real category, not imputed.

**`CrossAttentionBlock(dim, n_heads=8, mlp_ratio=4.0, dropout=0.1)`**
- Pre-norm `nn.MultiheadAttention(batch_first=True)` + FFN (GELU), residuals.
- Stores `self.last_attn` = head-averaged weights `(B, n_query, n_key)`, detached.

**`CrossAttentionFusion(image_encoder, tokenizer, dim, n_classes, n_blocks=2, n_heads=8, direction="img2meta", meta_field_dropout=0.2)`**
- `image_encoder` must return `(B, N_patches, dim)`. **No timm wrapper exists yet.** ConvNeXt
  feature maps need flattening + projection to `dim`.
- `direction ∈ {img2meta, meta2img, bidirectional}`; any other value raises `ValueError`.
- `_field_dropout_mask`: in training only, masks each field with probability `meta_field_dropout`
  via `key_padding_mask`. Field 0 is never masked. Only applied on the img2meta path.
- `forward(image, cat, cont, cont_mask) -> logits (B, n_classes)`.
- `attention_maps(grid_hw) -> (B, M, H, W)` or `None` (if there's no img2meta stack or no forward
  pass yet). Raises if `H*W != N`.

### `src/medfusion/eval/localization.py`

| Function | Does |
|---|---|
| `binarize(saliency, threshold=None)` | `saliency >= threshold`; default threshold = mean + std |
| `iou(pred_mask, gt_mask)` | IoU; NaN if the union is empty |
| `miou(preds, gts, threshold=None)` | Mean IoU over pairs, ignoring NaNs |
| `hit_rate(saliency, point_yx)` | True if argmax == the exact point (see §15) |
| `pointing_game(saliency, gt_mask)` | True if argmax falls inside the GT mask |
| `attention_mass_in_region(saliency, region_mask)` | Fraction of clipped-positive saliency inside the region (artifacts/drains: lower is better; pathology: higher is better) |

### `src/medfusion/eval/shortcut.py`

- `BIAS_LEVELS = [0, 1, 2, 3, 4, 5]` (index 5 = reversed correlation).
- `DegradationCurve(model, scores: dict[int, float])`: `.drop()` = score at lowest level − score
  at highest level; `.as_row()` → `{"model", "drop", "bias_0", ...}`.
- `drain_stratified_auc(y_true, y_score, drain_present)` → `{"drain_present", "drain_absent", "gap"}`
  (NaN when a stratum has one class).

### `src/medfusion/eval/stats.py`

- `mean_ci(values, alpha=0.05)` → `(mean, lo, hi)`, percentile bootstrap over 10,000 resamples, NaNs dropped.
- `paired_bootstrap(scores_a, scores_b, n=10_000, alpha=0.05)` → `(mean_diff, lo, hi, significant)`,
  where diff = b − a and "significant" means the CI excludes 0. Requires matched shapes.

### `scripts/prepare_padufes.py` (working)

- Finds `metadata.csv` recursively under `--raw`; requires columns `patient_id`, `img_id`, `diagnostic`.
- Uppercases labels and drops anything outside the 6 classes. Prints class distribution, patient
  count and biopsy rate (`biopsed` column).
- `StratifiedGroupKFold(n_splits=--n-folds [5], shuffle=True, random_state=--seed [0])` grouped by
  `patient_id` → writes a `fold` column.
- Writes `<out>/metadata.csv` and symlinks (or with `--copy-images`, copies) every `*.png` into
  `<out>/images/`.

### `scripts/train.py`: what must be built before it runs

1. `src/medfusion/data/padufes.py`: a Dataset returning `(image, cat, cont, cont_mask, label)`.
2. The CV loop: 5 folds × 3 seeds.
3. Per-fold metric collection feeding `eval/stats.mean_ci`.
4. **Do not add a single-split shortcut.**

### Tests (`pytest -q`, 13 pass)

- `test_fusion.py`: output shape for all 3 directions; attention maps are `(2, 5, 14, 14)` for
  img2meta; `None` for meta2img; a masked continuous value doesn't affect output; field dropout is
  active only in training.
- `test_eval.py`: IoU perfect/disjoint, mIoU, pointing game, attention-mass fraction, degradation
  drop, drain-stratified AUC doesn't crash.

---

## 10. Configs reference

Config style is Hydra-like YAML, but **no config loader exists yet** and Hydra isn't a dependency.

**`dataset_padufes20.yaml`:** root `data/processed/pad-ufes-20`, `images/`, `metadata.csv`; classes
`[BCC, SCC, ACK, SEK, MEL, NEV]`, label column `diagnostic`; patient-stratified 5-fold, seeds
[0,1,2], group `patient_id`; image 224, ImageNet mean/std.
- Categorical (18): `region, fitspatrick` (the CSV really spells it this way), `gender, smoke, drink,
  background_father, background_mother, pesticide, skin_cancer_history, cancer_history,
  has_piped_water, has_sewage_system, itch, grew, hurt, changed, bleed, elevation`.
- Continuous (3): `age, diameter_1, diameter_2`.
- ⚠️ Check these column names against the downloaded CSV before the first run.

**`dataset_nih_cxr14.yaml`:** root `data/processed/nih-cxr14`, `Data_Entry_2017.csv`; multilabel,
14 classes (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax,
Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia); group `Patient ID`;
categorical `Patient Gender, View Position`; continuous `Patient Age`.

**`model_crossattn.yaml` (M1):** `convnext_tiny` (ablation: `vit_base_patch16_224`), pretrained
`imagenet` (→ `isic2020` later), `freeze_stages: 2`, `dim: 768`, fusion `n_blocks: 2`, `n_heads: 8`,
`direction: img2meta`, `meta_field_dropout: 0.2`, head dropout 0.3.

**`model_baselines.yaml`:** B0 `image_only`; B1 `metadata_only` (gradient boosting); B2
`late_fusion` (logit mean); B3 `concat` at pooled embedding; B4 `film` on metadata embedding.

**`exp_derm_fusion.yaml` (D1):** 60 epochs, batch 32, AdamW, lr 3e-4, layer-wise LR decay 0.75,
wd 0.05, cosine schedule, 5 warmup epochs, early stopping on `cv_mean_balanced_accuracy` with
patience 12, cross-entropy with balanced class weights. Augmentation: hflip 0.5, vflip 0.5,
rotate 30, color jitter 0.4, random-resized-crop [0.7, 1.0], RandAugment n=2 m=9, mixup 0.2.
Report balanced_accuracy, macro_f1, per_class_auc with bootstrap 95% CI.

**`exp_derm_shortcut.yaml` (D3):** 7 artifacts; 2,594 ISIC + 872 Atlas images; bias levels
[0..5]; models `B0_image_only, B3_early_concat, M1_crossattn`; seeds [0,1,2]; primary
`auc_vs_bias_level`, secondary `attention_mass_in_artifact_regions`.

**`exp_cxr_localization.yaml` (X2):** train on NIH-CXR14; eval `chexpert_val_test`; CheXlocalize
10 pathologies + 3-radiologist benchmark; 7 saliency methods; metrics `miou, hit_rate,
pointing_game`; stratify by `pathology, lesion_area_quartile, shape_complexity`.

**`exp_cxr_shortcut.yaml` (X3):** NEATX (3,500 CXR14 drains, 1,000 PadChest tubes); target
Pneumothorax; strata drain_present/absent; primary `auc_per_stratum_and_gap`, secondary
`attention_mass_in_drain_regions`.

---

## 11. Work phases and milestones

Nine working weeks, starting when the venue decision (D-2) is made.

| Phase | Weeks | Work | Exit criterion |
|---|---|---|---|
| 0. Decide & provision | Week 0 (2 days) | Fix venue + page limit; close D-1; provision 200–250 GB + GPU; file PhysioNet application in the background | D-1 and D-2 closed in writing |
| 1. Acquire | Week 1 | ~62 GB instant batch; Redivis + 3 agreements (~13 GB); clone Bissoto repos; checksum + `SOURCE.md` for each | Every dataset on disk with size + checksum |
| 2. Preprocess | Week 2 | Resize 224² (384² for localization); harmonize 26 → 21 PAD fields → `data/processed/padufes20_meta.csv`; patient-stratified splits → `data/processed/splits/`; fill `label_mapping.md`; align SkinCon/Bissoto/CheXlocalize/NEATX annotations to image IDs | A dataloader returns (image, metadata tokens, label, annotation mask) for every track with no manual step |
| 3. Baselines | Weeks 3–4 | B0–B4 under the full protocol (5-fold × 3 seeds, 95% CI) | T1 baseline rows filled, reproducible. **Checkpoint:** if B3 already matches M1, this is where we find out |
| 4. Fusion model | Weeks 4–5 | M1 + D-4 direction ablation + D-3 backbone ablation, heavily regularized | T1 complete (expect ~+2.5%, don't headline it) |
| 5. XAI evaluation (**the paper**) | Weeks 5–7 | D4 → T3; D3 → F2; X2 → F3; X3 → F4; attention-mass secondary rows | Four quantitative XAI results with CIs and a stated failure mode each |
| 6. External validation & fairness | Week 7 | Zero-shot DDI, stratified by Fitzpatrick tone → T2 | T2 produced and reported whatever it shows |
| 7. Write & submit | Weeks 8–9 | Freeze F1–F5, T1–T3; abstract leads with localization + shortcut results; reproducibility appendix | Submitted; `experiments/` reproduces every number |

**Hard gates:** week 0 (venue + track), week 1 (data on disk), week 2 (dataloaders + label mapping),
week 7 (four XAI results + fairness table), week 9 (submitted).

---

## 12. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | PhysioNet credentialing misses the deadline | High | High → now harmless | Off the critical path; build on NIH-CXR14; make the MIMIC swap a config change |
| R2 | Accuracy gain ~2.5%, inside the noise | Near-certain | High if the paper depends on it | Base the paper on localization + shortcut results; report accuracy with CIs as modest |
| R3 | Cross-attention overfits PAD-UFES-20 | High | High | Patient-stratified 5-fold CV, heavy augmentation, metadata field dropout, frozen early blocks, ISIC-pretrained image branch, early stopping on the CV mean |
| R4 | Derm7pt access slow or refused | Medium | Medium | Already cut; SkinCon is the concept ground truth |
| R5 | Both tracks exceed the page limit | High | High | Derm primary; CXR only if pages allow |
| R6 | Label-space mismatch with external sets | Certain | Medium | Fill `label_mapping.md` first; report dropped classes |
| R7 | XAI metric gaming (good mIoU, uninformative map) | Medium | Medium | Report mIoU **and** hit-rate **and** most-representative-point; qualitative panel; failure cases |
| R8 | Compute (bias levels × models × seeds × folds) | Medium | Medium | Budget the grid up front; cut seeds on D3 before bias levels |
| R9 | Worse performance on darker skin tones | Medium | Medium, but publishable | Report it; a stratified negative result is a contribution |

---

## 13. Open decisions

| ID | Decision | Status | Recommendation |
|---|---|---|---|
| **D-1** | Derm, CXR or both? | **Closed 2026-09-17: dermatology only** | CXR (CheXlocalize, NIH-CXR14 view position, NEATX) moves to the journal extension |
| **D-2** | Target venue + deadline | **Closed 2026-09-17: MIDL 2027** (est. early Dec 2026) | Fallback MICCAI 2027 (est. late Feb 2027), then iMIMIC / MultiTab 2027. ISBI 2027 (26 Oct 2026, 4 pages) rejected as too small; SPIE MI 2027 closed |
| D-3 | Image backbone | Open (run as ablation) | ConvNeXt-T for derm (more sample-efficient); ViT-B/16 for CXR (enables attention rollout) |
| D-4 | Cross-attention direction | Open (run as ablation) | Run img2meta / meta2img / bidirectional; only img2meta gives spatial maps |
| D-5 | Which saliency methods | Open | Minimum: Grad-CAM + raw cross-attention + one gradient-free method; all 7 if pages allow |

**Asked of the supervisor (report, 8 Sep 2026):**
1. Target conference + page limit.
2. Skin only, or skin + CXR (~60 GB and ~2 weeks of compute difference).
3. Approval + reference for the PhysioNet credentialing application.
4. Confirmation of a ~250 GB volume and GPU access.

---

## 14. Rules and conventions

**Experiments** (`experiments/<experiment>_<NNN>/`) must contain:

```
config.yaml     exact config snapshot (a copy, not a reference)
git_sha.txt     commit the run was launched from, plus `git diff` if the tree is dirty
metrics.json    per-fold, per-seed raw metrics, never only the aggregate
log.txt
maps/           cached saliency maps for evaluate_xai.py
```

1. Never overwrite a run directory; a new attempt gets a new number.
2. A run not reproducible from `config.yaml` + `git_sha.txt` doesn't go in the paper.
3. Record failed and abandoned runs too; the count goes in the supplementary material.

**General**
- If an experiment doesn't produce a numbered artifact in `results/`, don't run it.
- `notebooks/` is for exploration only; nothing in the paper comes from there.
- Always split at the patient level.
- Nothing is reported without CIs across folds × seeds.
- Data never goes into git or any upload.
- Code style: ruff, line length 100; `src/` layout; pytest `pythonpath = ["src"]`.
- Git: main branch should be `main` (currently working on `master` with no commits).

**Label mapping rules** (`docs/label_mapping.md`)
- PAD-UFES-20 → Derm7pt / DDI tables are all TODO. ACK may have no counterpart. DDI is
  malignant/benign at the top level, so consider collapsing to binary.
- State in the paper exactly which classes were dropped and why.
- If external evaluation is binary, also report the in-domain binary number.
- PAD-UFES-20 is smartphone clinical; Derm7pt has clinical + dermoscopic. Test on the clinical
  images; report the dermoscopic images separately as a modality-shift result.

---

## 15. Known issues and TODOs in the code

Found while writing this handbook (2026-09-16). None have been fixed yet.

1. **`hit_rate` is effectively always False on real data.** `localization.py` checks
   `argmax == (y, x)` exactly. **Resolved protocol (2026-09-17):** CheXlocalize's hit rate *is* the
   pointing game (argmax inside the GT mask, no tolerance), so drop `hit_rate` in favour of
   `pointing_game`. There is no test for it.
2. **`binarize` docstring says "Otsu-style"**, but the code uses mean + std. **Implement Otsu**
   (CheXlocalize's default); keep mean + std only as a sensitivity analysis.
3. **`stats.mean_ci` / `paired_bootstrap` use the unseeded global `np.random`**, so CIs change from
   run to run. Pass an `np.random.default_rng(seed)`.
4. **Paired bootstrap vs DeLong:** the plan names DeLong for AUC comparisons and multiple-comparison
   correction. Neither is implemented.
5. **`test_drain_stratified_gap` checks almost nothing.** Its assertion is always true. Test a case
   where both strata have both classes.
6. **Bias-level naming is inconsistent.** Docs describe levels "0 → 1" (a correlation strength);
   code/config use indices `0..5`. **Resolved:** use the published values
   `BIAS_LEVELS = [0, 0.3, 0.5, 0.7, 0.9, 1.0]` and the 10 splits per level from
   `artifact-generalization-skin`.
7. **Metadata field dropout is not applied on the `meta2img` path** (only img2meta gets
   `key_padding_mask`), so the bidirectional/meta2img ablations are regularized differently.
8. **No image encoder wrapper.** `CrossAttentionFusion` expects `(B, N, dim)` tokens; timm ConvNeXt
   returns `(B, C, H, W)` with C=768 for convnext_tiny's last stage (works with `dim: 768` after
   flatten). ViT-B/16 returns a CLS token that must be dropped for a 14×14 grid.
9. **No config loader.** Configs reference other configs by path (`dataset: configs/...yaml`);
   something must resolve them.
10. `prepare_padufes.py` only globs `*.png`. Check the image extension in the actual download.

---

## 16. Reference library

**Gap statement:** Groups A and B fuse modalities and add attention; group C fuses images. **None
scores its attention against human annotation.** All of them report accuracy, and where there is an
explanation it is a qualitative figure.

### A. Metadata/tabular + image fusion: direct competitors

| # | Paper | Use |
|---|---|---|
| R1 | *Attention-Guided Clinical-Fusion with Diagnostic Refinement and eXplainability Using Clinical Data and Image Analytics for Smart Healthcare Diagnosis* (ICEECCOT) | Closest prior art; their XAI is qualitative, ours is scored |
| R2 | *Multi-Modal Fusion Network Integrating Imaging and Clinical Tabular Data for Alzheimer's Disease Classification* | Pattern is established outside derm |
| R3 | *Multi-Modal Diffusion Network for Medical Image Classification via Imaging and Clinical Data Fusion* (ICCTC) | Non-attention comparison point |
| R4 | *Hypergraph Neural Networks with Attention-based Fusion for Multimodal Medical Data Integration and Analysis* (ICIC) | "Attention isn't the only way to fuse" |
| R5 | *Research and Application of Multimodal Data Fusion in Medical Images* | Survey framing |
| R6 | *From Text to Diagnosis: Exploring LLM-Assisted Prompt Strategies for Multimodal Few-Shot Medical Image Learning* | LLM alternative; mention, don't implement |

### B. Cross-attention mechanisms: architectural lineage

| # | Paper | Use |
|---|---|---|
| R7 | *Conjugate Cross-Modal Attention UNet for Medical Image Segmentation* (ICEIB) | Nearest architectural neighbour |
| R8 | *UMCA: Enhancing UNet with Multi-Scale Cross Attention for Semi-Supervised Medical Image Fusion* | D-4 direction ablation |
| R9 | *BAF-Net: Bidirectional Attention-aware Fluid Pyramid Feature Integrated Multi-modal Fusion Network for Prognosis* | Precedent for the bidirectional variant |
| R10 | *Cross-modal Frequency-aware Transformer for Multimodal Medical Image Fusion* (ICIPMC) | Transformer cross-modal fusion |
| R11 | *SAC UW-Net: A Self-attention-based Network for Multimodal Medical Image Segmentation* | Why cross- rather than self-attention |
| R12 | *Multi-Stage Cross-Level Adaptive Information Transfer for Medical Image Segmentation* (ICIAIS) | Where in the encoder to fuse |
| R13 | *A Multimodal Medical Image Fusion Method Incorporating an Adaptive Attention Mechanism* (CSCWD) | Adaptive attention weighting |
| R14 | *FedResNet: A Lightweight Federated Medical Image Classification Network Combining Cross-Stage Attention and Edge Enhancement* | Efficiency framing only |
| R15 | *Fau-Net: Fourier Attention Based 3D U-Net for Medical Image Registration* | Peripheral |
| R16 | *ADFA: Attention-Augmented Differentiable Top-K Feature Adaptation for Unsupervised Medical Anomaly Detection* | "Which features does attention use" |

### C. Classical image-to-image fusion: scope boundary

R17 *Multimodal Medical Image Fusion using Redundant Discrete Wavelet Transform* (ICAPR) ·
R18 *A Recent Advancement in Multi-Modal Medical Image Fusion Techniques using SWT, NSST, NSCT, and CNN* (IC3ECSBHI) ·
R19 *A Comparative Analysis of Multi-Modal Medical Image Fusion Techniques using MSVD, WPD, PCA, and DWT* (IC3ECSBHI) ·
R20 *Development of Multimodal Fusion Technique for Medical Images* (ICAC3N) ·
R21 *GIMI: A New Evaluation Index for 3D Multimodal Medical Image Fusion* (CIS) ·
R22 *Multimodal Medical Image Segmentation Algorithm Based on Convolutional Neural Networks* (MIPR) ·
R23 *Multi-Domain Medical Image Enhancement and Diagnosis: A Unified Deep Generative Framework with Transferable Learning and Attention* ·
R24 *Multi-Domain Image Translation for Medical Imaging: A Novel Framework for Improved Modality Synthesis*

### Dataset citations (must be cited)

| Key | Source |
|---|---|
| D-PAD | PAD-UFES-20, Mendeley, DOI `10.17632/zr7vgbcyr2.1` |
| D-FITZ | Fitzpatrick17k |
| D-SKINCON | SkinCon |
| D-DDI | Diverse Dermatology Images |
| D-BISS | Bissoto et al., `github.com/alceubissoto/debiasing-skin` |
| D-ISIC | ISIC 2018 Task 1&2 training input |
| D-NIH | NIH ChestX-ray14 (Wang et al.) |
| D-CHEXPERT | CheXpert (Irvin et al.) |
| D-CHEXLOC | CheXlocalize (Saporta et al.) |
| D-NEATX | NEATX |

### Method citations

| Key | Method | Use |
|---|---|---|
| M-GRADCAM | Grad-CAM (Selvaraju et al.) | Primary saliency; CheXlocalize winner |
| M-ROLLOUT | Attention rollout (Abnar & Zuidema) | Second channel with a ViT backbone |
| M-FILM | FiLM (Perez et al.) | Baseline B4 |
| M-XATTN | Cross-attention (Vaswani et al.; Chen et al., CrossViT) | The fusion block |

Citation trail from the brief (convert to real references before submission): Mendeley Data ·
ScienceDirect · arXiv (Derm7pt, ISIC, NEATX, Bissoto) · ACM DL (SkinCon) · PhysioNet · Stanford AIMI
(CheXlocalize) · ResearchGate (PAD-UFES-20 metadata-gain figures).

---

## 17. Next actions checklist

**Setup / housekeeping**
- [ ] Make the first git commit and push to a **private** remote (§3.1).
- [ ] Run `pip install -e ".[dev]"` so timm, pandas, grad-cam and opencv are installed.
- [ ] Fix the NVIDIA driver/library mismatch (reboot) or move to a GPU machine.

**Decisions (week 0)**
- [ ] D-2: venue + page limit.
- [ ] D-1: derm-only or derm + CXR.
- [ ] Supervisor sign-off + reference → file the PhysioNet application in the background.
- [ ] Confirm ~250 GB storage + GPU.

**Data (week 1)**
- [ ] Download PAD-UFES-20 and run `prepare_padufes.py`; check the column names against the config.
- [ ] Download SkinCon, Fitzpatrick17k(+C), the Bissoto repos, ISIC 2018 Task 1&2 (and NIH-CXR14 +
      NEATX if doing CXR).
- [ ] Create a Redivis account and get DDI, CheXpert (downsampled) and CheXlocalize.
- [ ] Write `SOURCE.md` with a SHA256 for each; update `docs/02-access-checklist.md`.

**Code (weeks 1–3)**
- [ ] `src/medfusion/utils/`: seeding, config loader (resolve nested config paths), logging, run-dir
      creation (config snapshot + git SHA).
- [ ] `src/medfusion/data/padufes.py`: Dataset → `(image, cat, cont, cont_mask, label)`; vocab
      building with the MISSING index.
- [ ] Image encoder wrapper (timm ConvNeXt-T / ViT-B/16 → `(B, N, 768)` tokens, `freeze_stages`).
- [ ] Baselines B0–B4.
- [ ] `scripts/train.py` CV loop (5 folds × 3 seeds) → `metrics.json`.
- [ ] Fix the §15 issues (hit-rate tolerance and seeded bootstrap first).
- [ ] Fill in `docs/label_mapping.md`.
