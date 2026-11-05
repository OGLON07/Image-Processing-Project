# RSNA Knee Abnormality Detection AI Challenge (Group 2)

End-to-end, leak-free, reproducible machine learning pipeline for the **2026 RSNA Knee Abnormality Detection AI Challenge**.

This solution implements the university group assignment Plan of Action (`Group2_Plan_of_Action.pdf`) combined with key methodological innovations from the 33rd-place Silver Solution:
1. **Two-Level Attention Multi-Instance Learning (MIL)**: Instance-to-Series and Series-to-Study pooling over variable series counts (3–14 per study) without padding or discarding slices.
2. **2.5D Volumetric MRI Input**: 3-channel windowing (center slice ± 1 adjacent slice), central anatomical crop, and per-series 2nd–98th percentile intensity normalization.
3. **Multilingual Report Cross-Attention Fusion**: Pretrained transformer text token embeddings guide visual spatial attention maps during training, paired with text-modality dropout ($p=0.4$) and a learnable null token.
4. **Site-Shortcut Trap Branch**: A training-only debiasing branch that absorbs site-specific reporting habits (pseudo-site random effects $a[\text{site}, \text{finding}]$ and scanner acquisition parameters $g(\cdot)$), completely bypassed at test inference to eliminate distribution shift.
5. **Zero Data Leakage**: Automated unit tests assert 100% study-level and series-level containment within folds.
6. **Strict Image-Only Inference**: Automated unit tests assert test prediction NEVER accesses reports or text tokens.

---

## Repository Structure

```text
├── configs/
│   └── default_config.yaml     # Central hyperparameter configuration
├── src/
│   ├── __init__.py
│   ├── config.py               # Python config dataclass
│   ├── eda.py                  # Phase 0: Exploratory data analysis & site bias analysis
│   ├── split.py                # GroupKFold leak-free split by StudyInstanceUID
│   ├── dataset.py              # DICOM windowing, 2.5D stacking, report label extraction
│   ├── transforms.py           # Spatial MRI data augmentations
│   ├── metrics.py              # Vectorized Macro-ROC-AUC and multi-label Log Loss
│   ├── losses.py               # Weighted multi-label BCE & confidence-weighted soft loss
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py         # 2.5D visual feature extractor
│   │   ├── mil_pooling.py      # Level 1 & Level 2 attention MIL
│   │   ├── cross_attention.py  # Text-guided cross-attention with null token
│   │   ├── trap_branch.py      # Site shortcut debiasing trap branch
│   │   └── full_model.py       # Full multimodal model & image-only inference path
│   ├── train.py                # 5-fold training loop with checkpointing
│   ├── evaluate.py             # Image-only validation evaluator
│   ├── ablations.py            # Systematic 4-stage ablation matrix
│   └── inference.py            # Image-only submission generator
├── tests/
│   ├── test_no_leakage.py      # Asserts 0 StudyInstanceUID / series overlap across folds
│   ├── test_image_only_inf.py  # Asserts test pathway never inspects or requires text
│   └── test_model_pipeline.py  # Asserts dynamic series counts (3, 5, 14) and output shapes
├── artifacts/                  # Generated metrics, ablation tables, fold splits, checkpoints
├── submission.py               # Standalone competition submission entry point
├── README.md                   # Project documentation and reproduction guide
└── REPORT.md                   # Comprehensive technical report
```

---

## Quickstart & Reproduction

### 1. Run Phase 0 (Exploratory Data Analysis)
```bash
python src/eda.py
```
Outputs `artifacts/eda_summary.json` containing real label distributions, language breakdowns, and series plane statistics.

### 2. Generate Leak-Free 5-Fold Splits
```bash
python src/split.py
```
Outputs `artifacts/train_folds.csv`.

### 3. Run Automated Acceptance Unit Tests
```bash
python -m unittest tests/test_no_leakage.py
python -m unittest tests/test_image_only_inf.py
python -m unittest tests/test_model_pipeline.py
```

### 4. Run Phase 3 Ablations & Cross-Validation
```bash
python src/ablations.py
python src/train.py
```

### 5. Generate Competition Submission (Strict Image-Only)
```bash
python submission.py
```
Outputs `submission.csv` matching `sample_submission.csv` exactly.

