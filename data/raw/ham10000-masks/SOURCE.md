# HAM10000: lesion segmentation masks

| Field | Value |
|---|---|
| Dataset | Binary lesion masks for all 10,015 HAM10000 images (`HAM10000_segmentations_lesion_tschandl`) |
| Official source | Harvard Dataverse **v4**, DOI [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T), file `HAM10000_segmentations_lesion_tschandl.zip` (10.8 MB) |
| Copy used | Kaggle `tschandl/ham10000-lesion-segmentations` (uploaded by the dataset author, P. Tschandl) |
| Mirror version | Not recorded by the Kaggle CLI; zip last-modified 2020-07-02 |
| Downloaded | 2026-09-17 12:43–12:44, `scripts/download_data.sh` |
| File | `ham10000-lesion-segmentations.zip`, 10,766,207 bytes |
| SHA256 | `d202bafd346e4ce16477575e871bc574528a8d044a0fbf654910ed49224bec7b` |
| Licence | CC BY-NC 4.0 (part of the HAM10000 Dataverse record) |
| DUA / access | None; open download |
| Verification | `scripts/verify_data.py`: 10,015 masks, one for each HAM10000 image ID |

## Cite

HAM10000 (see `../ham10000/SOURCE.md`) and the paper that introduced the masks:

```bibtex
@article{tschandl2020humancomputer,
  author  = {Tschandl, Philipp and Rinner, Christoph and Apalla, Zoe and others},
  title   = {Human--computer collaboration for skin cancer recognition},
  journal = {Nature Medicine}, volume = {26}, number = {8}, pages = {1229--1234}, year = {2020},
  doi     = {10.1038/s41591-020-0942-0}
}
```
