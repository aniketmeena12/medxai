# HAM10000: images and metadata (official Harvard Dataverse)

| Field | Value |
|---|---|
| Dataset | HAM10000: 10,015 dermoscopic images, 7 classes, with `HAM10000_metadata.csv` (lesion_id, image_id, dx, dx_type, age, sex, localization, dataset) |
| Source | Harvard Dataverse **v4**, DOI [10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T) — the official record (file ids 3172585, 3172584, 4338392) |
| Downloaded | 2026-09-19 15:25–15:37, `scripts/download_data_official.sh` (Linux machine, ~1.6 MB/s) |
| Licence | **CC BY-NC 4.0** (non-commercial) |
| DUA / access | None; open download. Do not redistribute images |

| File | Bytes | MD5 | SHA256 |
|---|---|---|---|
| `HAM10000_images_part_1.zip` | 1,366,522,108 | `4639bfa73ab251610530a97c898e6e46` | `6202a599129c2efe81c25aea8586d422eeda8a4fae5df49dd7b7bbf956d419db` |
| `HAM10000_images_part_2.zip` | 1,403,566,547 | `da43d6cc50f6613013be07e8986b384b` | `9677f30118c21ecc667a147de369d89acb25efab046fd32f6b13ad677ce2d6d0` |
| `HAM10000_metadata.csv` | 690,218 | — | `de17ec44cb25ea9c3f6378ba18d69224c740afd62e86ec2edc6a894d6aa663d3` |

- **Both image MD5s match the values Dataverse publishes for those files** (checked 2026-09-19), so
  this is the official bitstream, not a mirror.
- `HAM10000_metadata.csv` is fetched with `?format=original`; Dataverse's own tabular conversion
  (`HAM10000_metadata.tab`) is not used. It has 8 columns — one more (`dataset`, the source cohort)
  than the Kaggle mirror's copy. Neither `dataset` nor `dx_type` is a model input
  (`configs/data_ham10000.yaml`): both leak how the label was obtained.
- **Verification** (`scripts/verify_data.py`, 2026-09-19): 10,015 unique `.jpg`; metadata has 10,015
  rows and every `image_id` has an image; `dx` ∈ {akiec, bcc, bkl, df, mel, nv, vasc}; class counts
  nv 6,705 · mel 1,113 · bkl 1,099 · bcc 514 · akiec 327 · vasc 142 · df 115; 7,470 lesion groups.
- **Earlier copy:** the 2026-09-17 download (Windows machine, now gone) used the Kaggle mirror
  `kmader/skin-cancer-mnist-ham10000` (`skin-cancer-mnist-ham10000.zip`, 5,582,914,511 bytes,
  SHA256 `9c3fffba9a8522470472d13216cf2d4df7f3634880e5ff4ba6f0ba52daf5ebad`) because Dataverse ran
  at ~37 KB/s from that network.

## Cite

```bibtex
@article{tschandl2018ham10000,
  author  = {Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  title   = {The {HAM10000} dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  journal = {Scientific Data}, volume = {5}, pages = {180161}, year = {2018},
  doi     = {10.1038/sdata.2018.161}
}
```
