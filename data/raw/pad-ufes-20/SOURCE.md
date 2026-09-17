# PAD-UFES-20: images (Kaggle mirror)

| Field | Value |
|---|---|
| Dataset | PAD-UFES-20: 2,298 smartphone clinical images, 1,641 lesions, 1,373 patients, 6 classes |
| Official source | Mendeley Data **V1**, published 2020-07-07, DOI [10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1) |
| Copy used | Kaggle mirror `mahdavi1202/skin-cancer` (the official Mendeley download ran at ~22 KB/s) |
| Mirror version | Not recorded by the Kaggle CLI; zip last-modified 2023-10-24 |
| Downloaded | 2026-09-17 12:30–12:43, `scripts/download_data.sh` |
| File | `skin-cancer.zip`, 3,599,534,454 bytes |
| SHA256 | `77972a627cae21dcf20ab249d6a474f754acc968d485df8da63550e07480ec80` |
| Licence | CC BY 4.0 (official record) |
| DUA / access | None; open download |
| Verification | `scripts/verify_data.py`: 2,298 unique `.png` files, and every `img_id` in the official metadata has an image. **We use metadata only from `../pad-ufes-20-official/`**; the mirror's CSV has the same values but different formatting |

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
