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

## Setup (Windows, as used on the dev machine)

The project lives at `C:\Projects\medfusion-xai` (outside OneDrive, so data is never cloud-synced).
All data is in `data/` (gitignored); the virtual environment is `C:\medfusion-venv`.

```bash
python -m venv C:/medfusion-venv
C:/medfusion-venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
C:/medfusion-venv/Scripts/python -m pip install -e ".[dev]"
C:/medfusion-venv/Scripts/python -m pytest -q
```

Set `MEDFUSION_DATA` to use a data root other than `data/`.

## Pipeline

```bash
bash scripts/download_data.sh                        # PAD-UFES-20, HAM10000 (+masks), ISIC 2019, Bissoto repos
python scripts/prepare_data.py --dataset padufes20   # cache 256px images, grouped 5-fold splits
python scripts/prepare_data.py --dataset ham10000
python scripts/prepare_data.py --dataset isic2019    # after ham10000 (attaches shared masks)
python scripts/train.py --data configs/data_padufes20.yaml --model M1
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
