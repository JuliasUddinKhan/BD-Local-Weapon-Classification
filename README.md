# BD-LW: Bangladeshi Local Weapon Image Dataset and Benchmark

> **Status:** this repository accompanies a manuscript currently **under review** at *Multimedia Tools and Applications*. It is shared so that reviewers and editors can access the dataset and check the results. The citation will be added after publication.

BD-LW is an image dataset of six traditional Bangladeshi implements that are used as weapons in local violence. This repository contains the notebooks that build the dataset split, train and evaluate all models, and produce every table and figure in the manuscript.

**Dataset:** [Mendeley Data link / DOI]
**Manuscript:** "[Paper title]", submitted to *Multimedia Tools and Applications*

## For reviewers

- **Dataset:** download it from the link above. [If the dataset is not yet public: reviewers can request access at [email]; access is granted for the review only.]
- **Fixed split:** `splits/manifest.csv` gives the exact test set and cross-validation folds used in the manuscript.
- **Results:** every table in the manuscript can be traced to a notebook (see the table below), and the CSV files behind the tables are in `results/`.
- **Supplementary material:** `docs/supplementary_material.pdf` (also submitted to the journal) lists which notebook produces each result in Section S4.

---

## Dataset at a glance

| Class | Local name | English | Images |
|---|---|---|---|
| 0 | Churi | Knife | 521 |
| 1 | Dao | Machete | 530 |
| 2 | Khor | Straight razor | 471 |
| 3 | Kodal | Hoe | 477 |
| 4 | Kural | Axe | 477 |
| 5 | Shabol | Iron crowbar | 474 |
| | | **Total** | **2,950** |

- **Single subset:** 1,776 images, one weapon per image
- **Multi subset:** 1,174 images, several weapons of the same class
- **Capture sources:** iPhone 14, Redmi Note 15, Samsung Galaxy M14, WhatsApp-shared
- **Fixed split:** a locked test set of 419 images and five cross-validation folds, built so that near-identical photos never cross the split

<p align="center">
  <img src="assets/sample_images.png" width="85%" alt="One example image per class">
</p>

---

## Repository structure

```
bd-lw-benchmark/
├── notebooks/
│   ├── 00_stage0_curation.ipynb
│   ├── 01_background_removal.ipynb
│   ├── 02_training_scenarios.ipynb
│   ├── 03_analysis.ipynb
│   ├── 04_paper_figures.ipynb
│   ├── 05_gradcam_same_architecture.ipynb
│   ├── 06_ablation_check_convnext_vit.ipynb
│   ├── 07_no_weapon_false_alarms.ipynb
│   ├── 08_annotation_validation.ipynb
│   ├── 09_robustness_5fold.ipynb
│   ├── 10_gradcam_all_models.ipynb
│   ├── 11_clip_zero_shot.ipynb
│   ├── 12_merge_and_statistics.ipynb
│   └── 13_misclassified_test_images.ipynb
├── splits/
│   ├── manifest.csv          # image_id, path, class, subset, source, split, fold, group
│   ├── class_map.json
│   ├── test.txt
│   └── fold0.txt ... fold4.txt
├── results/                  # CSV files behind the tables in the paper
├── assets/                   # images used in this README
├── docs/                     # supplementary_material.pdf
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Notebooks

All notebooks were run on Kaggle (one Tesla T4 GPU). Run them in this order; each one reads the outputs of the earlier ones, which you attach with **Add Input > Your Work** on Kaggle. The first cell of every notebook lists its inputs.

| # | Notebook | What it does | GPU | Time |
|---|---|---|---|---|
| 0 | Stage 0 curation | Standardizes images, removes near-duplicates, builds similarity groups and the fixed split | No | ~20 min |
| 1 | Background removal | Weapon masks and cut-outs with IS-Net | Yes | ~30 min |
| 2 | Training scenarios | Trains all models and preprocessing pipelines in three scenarios (all, single, multi) | Yes | several sessions |
| 3 | Analysis | Statistics, transfer, robustness, calibration, cost | Yes | ~1 h |
| 4 | Paper figures | Figures for the paper | No | ~5 min |
| 5 | Grad-CAM, same architecture | Grad-CAM across scenarios and pipelines | Yes | ~15 min |
| 6 | Ablation check | Repeats key ablations with ConvNeXt-Tiny and ViT-S/16 | Yes | ~3.5 h |
| 7 | Images without weapons | False alarms on 112 non-weapon images | Yes | ~10 min |
| 8 | Annotation validation | Second-annotator agreement (Cohen's kappa) | No | ~5 min |
| 9 | Robustness, 5 folds | Six corruptions at five levels, all fold models | Yes | ~30 min |
| 10 | Grad-CAM, all models | Attention on the weapon, baselines, ViT attention rollout | Yes | ~20 min |
| 11 | Zero-shot CLIP | CLIP baseline without training | Yes | ~20 min |
| 12 | Merge and statistics | Combines 9 to 11, corrected CV statistics, group sizes | No | ~5 min |
| 13 | Misclassified images | Every misclassified test image | No | ~2 min |

---

## Main results

| Model | Test macro-F1 (E1) | Mean macro-F1 under corruption | CPU time / image |
|---|---|---|---|
| ResNet-50 | 0.998 | 0.919 | 164 ms |
| EfficientNet-B0 | 1.000 | 0.787 | 56 ms |
| ConvNeXt-Tiny | 0.998 | 0.918 | 153 ms |
| ViT-S/16 | 1.000 | 0.965 | 181 ms |
| DINOv2 + logistic regression | 0.985 | n/a | 77 ms |

Clean accuracy is near perfect, but models trained on one image type fall to 0.54 to 0.86 on the other, and a model trained on one phone falls to 0.534 on the others. See the manuscript for details.

---

## Quick start

1. Download the dataset from [Mendeley Data link] and upload it to Kaggle as a private dataset.
2. Open `notebooks/00_stage0_curation.ipynb` on Kaggle, add the dataset as input and run all cells.
3. Continue with the notebooks in the order above.

To use the dataset with the official split, read `splits/manifest.csv`: train on images with `split == "trainval"` (use the `fold` column for cross-validation) and report results on `split == "test"`.

### Requirements

```
pip install -r requirements.txt
```

Main packages: Python 3.10+, PyTorch 2.x, timm 1.0, scikit-learn, OpenCV, pandas, matplotlib, open_clip_torch (Notebook 11 only).

---

## Citation

The manuscript is under review. A citation will be added here after publication. Until then, please do not redistribute the dataset or use it in other publications without contacting the authors.

---

## License

- **Code:** MIT License (see `LICENSE`)
- **Dataset:** CC BY-NC 4.0, for research use only. It must not be used for automated surveillance or profiling of people. During the review period, please do not redistribute it.

## Contact

[Corresponding author name], [email]
