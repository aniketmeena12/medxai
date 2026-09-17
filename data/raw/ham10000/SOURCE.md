# HAM10000: images and metadata (Kaggle mirror)

| Field | Value |
|---|---|
| Dataset | HAM10000: 10,015 dermoscopic images, 7 classes, with `HAM10000_metadata.csv` (lesion_id, dx, dx_type, age, sex, localization) |
| Official source | Harvard Dataverse **v4**, DOI [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T) |
| Copy used | Kaggle mirror `kmader/skin-cancer-mnist-ham10000` (Harvard Dataverse ran at ~37 KB/s) |
| Mirror version | Not recorded by the Kaggle CLI; zip last-modified 2019-10-06 |
| Downloaded | 2026-09-17 12:44–13:04, `scripts/download_data.sh` |
| File | `skin-cancer-mnist-ham10000.zip`, 5,582,914,511 bytes |
| SHA256 | `9c3fffba9a8522470472d13216cf2d4df7f3634880e5ff4ba6f0ba52daf5ebad` |
| Licence | **CC BY-NC 4.0** (non-commercial) |
| DUA / access | None; open download. Do not redistribute images |
| Verification | `scripts/verify_data.py`: 10,015 unique `.jpg`; metadata has 10,015 rows, all images present; `dx` ∈ {akiec, bcc, bkl, df, mel, nv, vasc} |

## Cite

```bibtex
@article{tschandl2018ham10000,
  author  = {Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  title   = {The {HAM10000} dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  journal = {Scientific Data}, volume = {5}, pages = {180161}, year = {2018},
  doi     = {10.1038/sdata.2018.161}
}
```
