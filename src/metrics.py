"""Evaluation Metrics for RSNA Knee Abnormality Detection.
Primary Metric: Macro-averaged ROC-AUC across all 12 abnormalities.
Secondary Metric: Multi-label Weighted Log Loss.
Fully vectorized with pure NumPy for zero-dependency portability and exactness.
"""

from typing import Dict, List, Tuple, Union
import numpy as np

def compute_roc_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes exact ROC-AUC for a single binary target using Mann-Whitney U statistic.
    Handles tied predictions and edge cases (all positives or all negatives).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Filter out NaNs if any
    valid_mask = ~np.isnan(y_true)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined/trivial baseline
        
    # Rank ordering (using average ranks for ties)
    order = np.argsort(y_pred)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_pred)) + 1
    
    # Handle ties
    sorted_pred = y_pred[order]
    unique_vals, inverse, counts = np.unique(sorted_pred, return_inverse=True, return_counts=True)
    if len(unique_vals) < len(y_pred):
        tie_ranks = np.bincount(inverse, weights=ranks[order]) / counts
        ranks = tie_ranks[inverse][np.argsort(order)]
        
    rank_sum_pos = np.sum(ranks[y_true == 1])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))

def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """Computes binary cross entropy / log loss."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    valid_mask = ~np.isnan(y_true)
    y_true = y_true[valid_mask]
    y_pred = np.clip(y_pred[valid_mask], eps, 1.0 - eps)
    
    if len(y_true) == 0:
        return 0.0
    loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return float(loss)

def compute_competition_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Computes primary macro-averaged ROC-AUC across all 12 abnormalities and individual AUCs,
    plus secondary weighted log loss.
    
    Args:
        y_true: Ground truth array of shape (N, 12).
        y_pred: Predicted probabilities array of shape (N, 12).
        target_cols: List of 12 finding names.
    Returns:
        dict with:
            - "macro_auc": Primary competition evaluation metric
            - "finding_aucs": Dict of individual AUCs for each abnormality
            - "weighted_log_loss": Multi-label cross entropy
            - "finding_log_losses": Dict of individual log losses
    """
    if target_cols is None:
        target_cols = [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA", "Effusion",
            "Synovitis", "Baker's", "Contusion", "Fracture"
        ]
        
    num_targets = y_true.shape[1]
    finding_aucs = {}
    finding_losses = {}
    
    for i in range(num_targets):
        col_name = target_cols[i] if i < len(target_cols) else f"target_{i}"
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        
        auc = compute_roc_auc_score(col_true, col_pred)
        loss = compute_log_loss(col_true, col_pred)
        
        finding_aucs[col_name] = round(auc, 4)
        finding_losses[col_name] = round(loss, 4)
        
    macro_auc = float(np.mean(list(finding_aucs.values())))
    mean_log_loss = float(np.mean(list(finding_losses.values())))
    
    return {
        "macro_auc": round(macro_auc, 4),
        "mean_log_loss": round(mean_log_loss, 4),
        "finding_aucs": finding_aucs,
        "finding_log_losses": finding_losses
    }

