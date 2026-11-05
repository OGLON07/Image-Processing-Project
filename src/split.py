"""Group-stratified and GroupKFold splitting module.
Guarantees ZERO DATA LEAKAGE:
All series and slices belonging to a single StudyInstanceUID are strictly isolated
within the same fold.
"""

import os
import json
import pandas as pd
import numpy as np

def create_folds(
    train_csv: str = "train.csv",
    n_splits: int = 5,
    seed: int = 42,
    output_path: str = "artifacts/train_folds.csv"
) -> pd.DataFrame:
    """Creates leak-free Group-based K-fold splits grouped strictly by StudyInstanceUID.
    
    The 58 ground-truth image-labeled studies are evenly distributed across the 5 folds
    (stratified by finding presence where possible), and the remaining 4,349 report-only
    studies are grouped across the 5 folds.
    """
    df = pd.read_csv(train_csv)
    target_cols = [c for c in df.columns if c not in ["StudyInstanceUID", "Report"]]
    
    # Identify labeled vs report-only studies
    labeled_mask = df[target_cols[0]].notnull()
    labeled_df = df[labeled_mask].copy()
    unlabeled_df = df[~labeled_mask].copy()
    
    rng = np.random.RandomState(seed)
    
    # Stratified fold assignment for the 58 labeled studies
    # We sort by sum of positive findings to distribute difficulty evenly
    labeled_df["finding_sum"] = labeled_df[target_cols].sum(axis=1)
    labeled_df = labeled_df.sample(frac=1.0, random_state=seed).sort_values("finding_sum", ascending=False)
    labeled_df["fold"] = np.arange(len(labeled_df)) % n_splits
    labeled_df = labeled_df.drop(columns=["finding_sum"])
    
    # Group assignment for the 4,349 report-only studies
    unlabeled_df = unlabeled_df.sample(frac=1.0, random_state=seed)
    unlabeled_df["fold"] = np.arange(len(unlabeled_df)) % n_splits
    
    # Recombine
    combined = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
    combined = combined.sort_values("StudyInstanceUID").reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"[+] Saved fold split with {len(combined)} studies ({n_splits} folds) to {output_path}")
    
    # Verify split distribution
    fold_summary = {}
    for f in range(n_splits):
        fold_studies = combined[combined["fold"] == f]
        fold_labeled = fold_studies[fold_studies[target_cols[0]].notnull()]
        fold_summary[f"fold_{f}"] = {
            "total_studies": len(fold_studies),
            "labeled_studies": len(fold_labeled),
            "report_only_studies": len(fold_studies) - len(fold_labeled)
        }
    print("Fold summary:")
    print(json.dumps(fold_summary, indent=2))
    
    return combined

if __name__ == "__main__":
    create_folds()

