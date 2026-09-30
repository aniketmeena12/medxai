# MedFusion-XAI

**Same Accuracy, Different Evidence: does patient metadata ground or distract skin lesion
classifiers?** Target: MIDL 2027.

When a classifier is given patient metadata, does its image evidence become better grounded on the
lesion, or does the model stop looking at the image? We compare image-only and four fusion models
(concat, FiLM, MetaBlock, cross-attention) on lesion-mask alignment, artifact and metadata shortcut
trap sets, and a counterfactual metadata test.

- Plan and hypotheses (frozen before training): [`docs/04-preregistration.md`](docs/04-preregistration.md)
- Novelty and positioning: [`docs/research/03-novelty-and-direction.md`](docs/research/03-novelty-and-direction.md)
- Full project context: [`PROJECT_HANDBOOK.md`](PROJECT_HANDBOOK.md)

## Setup

Current machine (Linux, RTX 5080). The system Python is 3.14, which torch has no wheels for, so the
venv is pinned to 3.12:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q          # 42 passed
```

Earlier dev machine (Windows, RTX 3050): project at `C:\Projects\medfusion-xai`, venv
`C:\medfusion-venv`, same `pip install -e ".[dev]"` after the cu128 torch wheel.

Set `MEDFUSION_DATA` to use a data root other than `data/`; otherwise run the scripts from the
repository root.

## Pipeline

```bash
# 1. data (~19 GB download, 29 GB on disk once cached)
bash scripts/download_data_official.sh     # official: Dataverse (HAM10000), ISIC S3, Mendeley, GitHub
bash scripts/download_padufes_images.sh    # PAD-UFES-20 images: HF mirror; TARGET=kaggle for Kaggle
# bash scripts/download_data.sh            # alternative: all Kaggle mirrors (needs ~/.kaggle)
.venv/bin/python scripts/verify_data.py    # 15/15 checks against the official facts

# 2. processing: cache 256px images/masks, write grouped 5-fold splits
.venv/bin/python scripts/prepare_data.py --dataset padufes20
.venv/bin/python scripts/prepare_data.py --dataset ham10000
.venv/bin/python scripts/prepare_data.py --dataset isic2019   # after ham10000 (attaches shared masks)

# 3. tuning (pre-registered 6-trial grid per model), then the full protocol
bash scripts/queue_tune.sh padufes20
.venv/bin/python scripts/train.py --data configs/data_padufes20.yaml --model M1
.venv/bin/python scripts/train_trap.py --model M1 --curve      # ISIC 2019 trap sets
```

Models: `B0` image-only · `B1` metadata-only · `B3` concat · `B4` FiLM · `B5` MetaBlock · `M1` cross-attention.

## Layout

```
configs/            base.yaml + one file per dataset
src/medfusion/
  data/             metadata encoding (MISSING as a category), grouped splits, datasets, trap sets
  models/           ImageEncoder (timm), MetadataTokenizer, B0/B3/B4/B5/M1, B1 tabular
  train/            train_one + run_cv (all folds x seeds, raw per-fold results)
  xai/              Grad-CAM, RISE, parameter-randomization sanity gate
  eval/             lesion energy / pointing game / Otsu IoU, shortcut curves, counterfactual shift, stats
scripts/            download, prepare, train
docs/               research notes and the pre-registration
tests/
```

## Rules

- Data never goes into git or any upload (dataset licences prohibit redistribution).
- Splits are grouped by patient/lesion; nothing is reported without CIs across folds x seeds.
- Each run gets a new numbered directory with its config snapshot and git SHA.
