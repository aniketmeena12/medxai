# DDI: Diverse Dermatology Images (dataset D, external test only)

| Field | Value |
|---|---|
| Dataset | 656 biopsy-proven clinical photographs, binary malignant/benign, balanced across Fitzpatrick skin-tone groups (I-II / III-IV / V-VI; FST V-VI: 159 benign + 48 malignant) |
| Source | Stanford AIMI, DOI [10.71718/kqee-3z39](https://doi.org/10.71718/kqee-3z39) · Redivis `https://stanford.redivis.com/datasets/3r16-5mby7gfer` · canonical page `https://ddi-dataset.github.io/` |
| Size | ~239 MB |
| Licence / DUA | **Stanford University School of Medicine Research Use Agreement.** Personal, non-commercial research only; no redistribution or publication of the images; **each user must register individually** |
| Status | **NOT DOWNLOADED.** Requires a person to register and accept the agreement; it cannot be fetched programmatically |
| Role | External, zero-shot test of models trained on datasets A and B: skin-tone fairness, and an independent test of the offloading result (fusion models receive all-MISSING metadata) |
| Label mapping | Frozen in `docs/label_mapping.md` before download |

## To obtain it

1. Open the Redivis link above, sign in or register, and accept the Research Use Agreement.
2. Download the archive and place it (unmodified) in this folder, `data/raw/ddi/`.
3. `python scripts/prepare_data.py --dataset ddi` - it locates the metadata CSV by its columns
   (`DDI_file`, `malignant`, `skin_tone`), caches images at 256px and writes
   `data/processed/ddi/samples.csv`. No fold column: this set is only ever a test set.
4. Record the archive's SHA256 and the download date in this file.

## Cite

```bibtex
@article{daneshjou2022ddi,
  author  = {Daneshjou, Roxana and Vodrahalli, Kailas and Novoa, Roberto A. and others},
  title   = {Disparities in dermatology {AI} performance on a diverse, curated clinical image set},
  journal = {Science Advances}, volume = {8}, number = {31}, pages = {eabq6147}, year = {2022},
  doi     = {10.1126/sciadv.abq6147}
}
```
