# Results

CSV files behind the tables and figures of the manuscript. Each folder lists the notebook that produces it.

| Folder | Notebook | Contents |
|---|---|---|
| `tables/` | nb3-analysis-scenarios | Main analysis tables: dataset counts (T1), scenarios (T7), transfer between single and multi images (T8), capture-source tests (T9, T10), robustness of the first-fold models (T11), calibration (T12), speed and size (T13), DINOv2 baseline (T4b) |
| `gradcam_scenarios/` | nb5-gradcam-architecture | Grad-CAM share on the weapon for ConvNeXt-Tiny and ViT-S/16, three pipelines and scenarios (Supplementary Table S20) |
| `ablation_check/` | nb6-ablation-check | Key ablations repeated with ConvNeXt-Tiny and ViT-S/16 (Supplementary Table S10) |
| `no_weapon/` | nb7-no-weapon | Predictions and confidence on 112 images without weapons (Supplementary Table S11) |
| `annotation_validation/` | nb8-annotation-validation | Second-annotator agreement and disagreements (Supplementary Table S17) |
| `revision/robustness_5fold/` | nb9-robustness-5fold, nb12-merge-and-statistics | Robustness of all five fold models, corruption parameters (Fig. 6, Supplementary Table S16) |
| `revision/gradcam_all_models/` | nb10-gradcam-all-models | Grad-CAM of all four models, chance baselines, ViT attention rollout (Table 3, Supplementary Table S19) |
| `revision/clip_zero_shot/` | nb11-clip-timing-checkpoints | Zero-shot CLIP results and timm checkpoint names (Supplementary Table S21) |
| `revision/statistics/` | nb12-merge-and-statistics | Corrected cross-validation statistics and similarity-group sizes (Supplementary Tables S15, S22) |
| `revision/misclassified/` | nb13-misclassified-test-images | Every misclassified test image (Supplementary Fig. S10b) |
