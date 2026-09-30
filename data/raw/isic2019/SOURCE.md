# ISIC 2019: training images, ground truth, metadata

| Field | Value |
|---|---|
| Dataset | ISIC 2019 Challenge training set: 25,331 dermoscopic images, 8 diagnostic classes; image substrate for the Bissoto trap sets (`../../annotations/artifact-generalization-skin`) |
| Official source | ISIC Challenge 2019, https://challenge.isic-archive.com/data/#2019 |
| Images | **Official ISIC S3**: `https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip` (the 2026-09-17 download used the Kaggle mirror `kioriaanthony/isic-2019-training-input`, `isic-2019-training-input.zip`, 9,766,639,904 bytes, SHA256 `ce6b334373e15a2677103115292016ded1215e49b8bc752d96c06f72b9bda195`) |
| CSVs | Official ISIC S3: `https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_{GroundTruth,Metadata}.csv` |
| Downloaded | 2026-09-19 15:25–16:05 (images 40 min at ~3.9 MB/s), `scripts/download_data_official.sh`; 2026-09-17 on the previous machine |
| Licence | CC BY-NC 4.0 (ISIC 2019 challenge data) |
| DUA / access | None; open download. Do not redistribute images |
| Verification | `scripts/verify_data.py` (2026-09-19, 15/15 checks pass): 25,331 unique images; every ground-truth image present; all 20,599 trap-set image IDs present and each has an ISIC metadata row (`sex` known for 98.2%) |

| File | Bytes | SHA256 |
|---|---|---|
| `ISIC_2019_Training_Input.zip` (official) | 9,771,618,190 | `5075020b720c8f1b9b7f3ff85326c55ce0435e5906c41fbd338994feef886df1` |
| `ISIC_2019_Training_GroundTruth.csv` | 1,291,479 | `aa88e9638fe4df9ef330dc4ba22fa4bd475c44af692fffb870c73091ab25cdcd` |
| `ISIC_2019_Training_Metadata.csv` | 1,214,351 | `d93994a8ed201d474de9a7af7e17ec30929cbc4a5220659ecbde17cbe83e9316` |

## Cite (as required by the ISIC 2019 challenge)

```bibtex
@article{tschandl2018ham10000,
  author = {Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  title  = {The {HAM10000} dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  journal = {Scientific Data}, volume = {5}, pages = {180161}, year = {2018}, doi = {10.1038/sdata.2018.161}
}
@inproceedings{codella2018isic2017,
  author    = {Codella, Noel C. F. and Gutman, David and Celebi, M. Emre and others},
  title     = {Skin lesion analysis toward melanoma detection: A challenge at the 2017 {ISBI}, hosted by the {ISIC}},
  booktitle = {2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018)},
  pages = {168--172}, year = {2018}, doi = {10.1109/ISBI.2018.8363547}
}
@misc{combalia2019bcn20000,
  author = {Combalia, Marc and Codella, Noel C. F. and Rotemberg, Veronica and others},
  title  = {{BCN20000}: Dermoscopic Lesions in the Wild},
  year   = {2019}, eprint = {1908.02288}, archivePrefix = {arXiv}
}
```

## Notes (2026-09-19)

- Both CSVs have the **same SHA256 as the 2026-09-17 download** (`aa88e963…`, `d93994a8…`), so the
  official CSVs are unchanged.
- The official CSVs name 2,074 images `ISIC_xxx_downsampled`; the trap sets use the plain id.
  `prepare_data.py` keeps the file stem for lookup and the plain id everywhere else.
- Processed (`data/processed/isic2019/samples.csv`): 25,331 rows, 13,931 `lesion_id` groups; class
  counts NV 12,875 · MEL 4,522 · BCC 3,323 · BKL 2,624 · AK 867 · SCC 628 · VASC 253 · DF 239;
  HAM10000 lesion masks attached to the 10,015 shared images (9,083 of which are in the trap sets).
