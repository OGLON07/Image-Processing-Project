"""Configuration module for RSNA Knee Abnormality Detection (Group 2).
Centralizes all hyperparameters, file paths, model configurations, and random seeds.
"""

from dataclasses import dataclass, field
from typing import List
import os
import torch

@dataclass
class Config:
    # Reproducibility
    seed: int = 42
    
    # Paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv: str = os.path.join(base_dir, "train.csv")
    train_series_csv: str = os.path.join(base_dir, "train_series.csv")
    sample_sub_csv: str = os.path.join(base_dir, "sample_submission.csv")
    artifacts_dir: str = os.path.join(base_dir, "artifacts")
    
    # Target Labels (12 abnormalities defined by RSNA challenge)
    target_cols: List[str] = field(default_factory=lambda: [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture"
    ])
    
    # Anatomical planes
    planes: List[str] = field(default_factory=lambda: ["Sagittal", "Coronal", "Axial"])
    
    # Preprocessing / Image parameters
    img_size: int = 256
    slices_per_window: int = 3  # 2.5D stacking (center, -1, +1)
    norm_pct_low: float = 2.0   # 2nd percentile
    norm_pct_high: float = 98.0 # 98th percentile
    
    # Model Architecture
    backbone_name: str = "coatnet"  # CoAtNet or resnet/convnext
    in_channels: int = 3           # 2.5D window (3 slices)
    feature_dim: int = 512
    mil_hidden_dim: int = 256
    num_classes: int = 12
    
    # Multilingual Report Text Modeling
    text_model_name: str = "xlm-roberta-base"
    text_dim: int = 768
    text_dropout_prob: float = 0.4  # Text modality dropout during training
    use_cross_attention: bool = True
    
    # Training
    num_folds: int = 5
    batch_size: int = 4
    grad_accumulation_steps: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 6
    use_amp: bool = True
    
    # Loss weights
    aux_loss_weight: float = 0.5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

default_config = Config()

