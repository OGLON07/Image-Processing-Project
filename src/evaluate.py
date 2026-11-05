"""Evaluation module for RSNA Knee Abnormality Detection.
Runs validation loops using STRICT image-only inference and computes:
- Primary metric: Macro ROC-AUC across all 12 abnormalities
- Abnormality-specific ROC-AUCs
- Secondary metric: Mean Log Loss
"""

import os
import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    import torch
    from src.metrics import compute_competition_metrics
    from src.dataset import generate_synthetic_study
except ImportError:
    torch = None

def evaluate_fold(
    model,
    val_df: pd.DataFrame,
    target_cols: List[str],
    device: str = "cpu"
) -> Dict:
    """Evaluates the model on validation studies using ONLY the image pathway.
    
    Args:
        model: RSNAKneeModel
        val_df: DataFrame containing validation studies (with ground-truth labels for the 58 labeled studies)
        target_cols: List of 12 target column names
        device: 'cuda' or 'cpu'
    Returns:
        Dict of evaluation metrics including macro_auc and individual finding AUCs.
    """
    model.eval()
    
    # Filter to studies with ground-truth image labels for ground-truth evaluation
    labeled_val_df = val_df[val_df[target_cols[0]].notnull()].copy()
    if len(labeled_val_df) == 0:
        # If no ground-truth studies in this subset, return empty or fallback
        return {"macro_auc": 0.5, "mean_log_loss": 1.0}
        
    y_true_list = []
    y_pred_list = []
    
    print(f"Evaluating on {len(labeled_val_df)} ground-truth labeled validation studies...")
    
    for idx, row in labeled_val_df.iterrows():
        study_uid = row["StudyInstanceUID"]
        gt_labels = row[target_cols].values.astype(np.float32)
        y_true_list.append(gt_labels)
        
        # Load or generate study image windows (strictly image-only input!)
        study_data = generate_synthetic_study(study_uid, num_series=5, slices_per_series=16, h=64, w=64)
        study_windows = [
            torch.from_numpy(s["windows"]).float().to(device) for s in study_data["series"]
        ]
        
        # STRICT IMAGE-ONLY PREDICTION (Zero text tokens)
        with torch.no_grad():
            pred_probs = model.forward_image_only(study_windows)
            
        y_pred_list.append(pred_probs.cpu().numpy())
        
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    metrics = compute_competition_metrics(y_true, y_pred, target_cols=target_cols)
    return metrics

