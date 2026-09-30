# HAM10000: lesion segmentation masks (official Harvard Dataverse)

| Field | Value |
|---|---|
| Dataset | Binary lesion masks for all 10,015 HAM10000 images (`HAM10000_segmentations_lesion_tschandl`), dermatologist-corrected |
| Source | Harvard Dataverse **v4**, DOI [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T), file id 3838943 — the official record |
| Downloaded | 2026-09-19 15:37, `scripts/download_data_official.sh` |
| File | `HAM10000_segmentations_lesion_tschandl.zip`, 10,808,743 bytes |
| MD5 | `6e8d252e09cfdb0189199f15985a5b84` — **matches the value Dataverse publishes** (checked 2026-09-19) |
| SHA256 | `6d88d7df8cc806c938932939d50c17fb9ad206c143343e4702c81cf3453c1e7a` |
| Licence | CC BY-NC 4.0 (part of the HAM10000 Dataverse record) |
| DUA / access | None; open download |
| Verification | `scripts/verify_data.py` (2026-09-19): 10,015 masks, exactly one per HAM10000 image ID |

- The zip was made on macOS, so it also carries a `__MACOSX/` tree of AppleDouble stubs — including
  one 233-byte `._ISIC_0025504_segmentation.png` that looks like an 10,016th mask. `scripts/`
  ignores `__MACOSX/` and `._*` entries (`is_junk`); the real mask count is 10,015.
- Masks are cached to `data/interim/ham10000/masks_256/` with NEAREST resampling (values stay {0,255}).
- **Earlier copy:** the 2026-09-17 download used the Kaggle mirror `tschandl/ham10000-lesion-segmentations`
  (uploaded by the dataset author; `ham10000-lesion-segmentations.zip`, 10,766,207 bytes,
  SHA256 `d202bafd346e4ce16477575e871bc574528a8d044a0fbf654910ed49224bec7b`) — same masks, repacked
  without the macOS stubs.

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
