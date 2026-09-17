# Bissoto artifact annotations and trap sets (GitHub)

Both repos were shallow-cloned by `scripts/download_data.sh` on 2026-09-17 ~13:42. Each commit is
pinned in `<repo>.commit.txt`. Git records the files inside each repo, so no separate SHA256 is kept.

| Repo | Commit (HEAD) | Commit date | Used for |
|---|---|---|---|
| [`alceubissoto/artifact-generalization-skin`](https://github.com/alceubissoto/artifact-generalization-skin) | `54b20fbcd53524616f64bb5b9237eebc2cb270b8` | 2022-10-21 | **Primary.** Trap sets at bias levels {0, 0.3, 0.5, 0.7, 0.9, 1} × 10 splits on ISIC 2019; model-inferred artifact probabilities in `isic_inferred_wocarcinoma.csv` (threshold 0.6) |
| [`alceubissoto/debiasing-skin`](https://github.com/alceubissoto/debiasing-skin) | `f93c8654f50c581544f32ad6b1450cb925a176ff` | 2020-06-15 | Manual annotations of 7 artifacts (ISIC 2018 / Atlas); reference only |

- **Licence:** no licence file in either repo (UNVERIFIED; see `docs/research/01-dataset-verification.md`). Use for research only, cite both papers, and do not redistribute.
- **Images:** not included. The trap sets point to ISIC 2019 images (`../raw/isic2019/`).
- **Re-check:** `git -C data/annotations/<repo> rev-parse HEAD` must match the commit above.

## Cite

```bibtex
@inproceedings{bissoto2022artifact,
  author    = {Bissoto, Alceu and Barata, Catarina and Valle, Eduardo and Avila, Sandra},
  title     = {Artifact-Based Domain Generalization of Skin Lesion Models},
  booktitle = {Computer Vision -- ECCV 2022 Workshops}, series = {LNCS}, publisher = {Springer},
  year      = {2023}, doi = {10.1007/978-3-031-25069-9_10}
}
@inproceedings{bissoto2020debiasing,
  author    = {Bissoto, Alceu and Valle, Eduardo and Avila, Sandra},
  title     = {Debiasing Skin Lesion Datasets and Models? Not So Fast},
  booktitle = {2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages     = {3192--3201}, year = {2020}, doi = {10.1109/CVPRW50498.2020.00378}
}
```
