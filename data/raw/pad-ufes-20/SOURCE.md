# PAD-UFES-20: images (Hugging Face mirror)

| Field | Value |
|---|---|
| Dataset | PAD-UFES-20: 2,298 smartphone clinical images, 1,641 lesions, 1,373 patients, 6 classes |
| Official source | Mendeley Data **V1**, published 2020-07-07, DOI [10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1) |
| Copy used | Hugging Face `SalmaneExploring/pad-ufes-20`, revision `086dd94836d7b2094132a21406a75dcb6bf7ec2c` (last modified 2026-04-28), declared licence `cc-by-4.0` |
| Why a mirror | Mendeley's public API lists only `metadata.csv` at the record root and exposes no direct link for the image folders; its browser download ran at ~22 KB/s. No Kaggle credentials exist on this machine |
| Downloaded | 2026-09-19 15:33–15:41, `scripts/download_data_official.sh` route 2 (`scripts/download_padufes_images.sh`) |
| Layout | `hf-SalmaneExploring-pad-ufes-20/all_images/imgs_part_{1,2,3}/` — the original Mendeley filenames and three-part split (911 / 659 / 728 = 2,298) |
| On disk | 3.4 GB, 2,298 `.png`; per-file SHA256 in `SHA256SUMS-images.txt` (digest of that manifest: `4d5f2a0bb5469d05f59e70c82ccbba4e878a11b70a9397ad14cd5a528ec1310e`) |
| Licence | CC BY 4.0 (official record; the mirror declares the same) |
| DUA / access | None; open download. Do not redistribute images |

**Fidelity checks run before use (2026-09-19)** — the mirror is unofficial, so it is checked against
the official record rather than trusted:

1. Exactly **2,298** `.png` files, and **every `img_id` in the SHA256-verified official Mendeley
   `metadata.csv` has an image** (`scripts/verify_data.py`).
2. The mirror also ships a `metadata.csv`; loaded side by side with the official file it is
   **identical** (same 26 columns, same 2,298 rows, `DataFrame.equals` after sorting by `img_id`).
   It is still **not used** — only `../pad-ufes-20-official/metadata.csv` is.
3. Image sizes vary per file (e.g. 609², 394²), as expected for multi-device smartphone capture;
   a re-encoded or resized mirror would show one uniform size.

**Not yet ruled out:** byte-level re-encoding of the PNGs themselves cannot be checked without the
official images, since Mendeley publishes no per-file checksums. To close that gap, fetch the Kaggle
mirror recorded below (`TARGET=kaggle bash scripts/download_padufes_images.sh`, needs
`~/.kaggle/kaggle.json`) and diff the two copies per file.

**Earlier copy:** the 2026-09-17 download (Windows machine, now gone) used the Kaggle mirror
`mahdavi1202/skin-cancer` (`skin-cancer.zip`, 3,599,534,454 bytes,
SHA256 `77972a627cae21dcf20ab249d6a474f754acc968d485df8da63550e07480ec80`).

## Cite

```bibtex
@article{pacheco2020padufes20,
  author  = {Pacheco, Andre G. C. and Lima, Gustavo R. and Salom{\~a}o, Amanda S. and Krohling, Breno and others},
  title   = {{PAD-UFES-20}: A skin lesion dataset composed of patient data and clinical images collected from smartphones},
  journal = {Data in Brief}, volume = {32}, pages = {106221}, year = {2020},
  doi     = {10.1016/j.dib.2020.106221}
}
@misc{pacheco2020padufes20data,
  author = {Pacheco, Andre G. C. and others},
  title  = {{PAD-UFES-20}: a skin lesion dataset composed of patient data and clinical images collected from smartphones},
  howpublished = {Mendeley Data, V1}, year = {2020}, doi = {10.17632/zr7vgbcyr2.1}
}
```
