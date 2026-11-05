"""Ablation studies for RSNA Knee Abnormality Detection.
Compares:
1. Image-only Baseline (No text cross-attention, no aux labels, no site trap)
2. + Text-Guided Cross-Attention Fusion
3. + Auxiliary Report Labels (Soft confidence-weighted supervision)
4. + Site Shortcut Trap (Full Silver Solution recipe with pseudo-site debiasing)
Saves results to artifacts/ablation_table.csv and artifacts/ablation_table.json.
"""

import os
import json
import pandas as pd
import numpy as np

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
# Add src directory to path for direct import
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import torch
from train import train_one_fold


def run_ablations(
    folds_path: str = "artifacts/train_folds.csv",
    epochs: int = 2,
    device: str = "cpu",
    output_dir: str = "artifacts"
) -> pd.DataFrame:
    """Executes the systematic ablation matrix on Fold 0.
    """
    os.makedirs(output_dir, exist_ok=True)
    folds_df = pd.read_csv(folds_path)
    target_cols = [c for c in folds_df.columns if c not in ["StudyInstanceUID", "Report", "fold"]]

    experiments = [
        {
            "name": "1. Image-Only Baseline",
            "cross_attn": False,
            "aux_labels": False,
            "trap": False,
            "description": "Standard 2D/3D visual backbone + Two-Level MIL (trained on 58 labeled studies only)"
        },
        {
            "name": "2. + Text-Guided Cross-Attention",
            "cross_attn": True,
            "aux_labels": False,
            "trap": False,
            "description": "Visual backbone + text-guided cross-attention with XLM-R embeddings & null token"
        },
        {
            "name": "3. + Auxiliary Report Labels",
            "cross_attn": True,
            "aux_labels": True,
            "trap": False,
            "description": "Cross-attention + confidence-weighted soft BCE loss on 4,349 multilingual reports"
        },
        {
            "name": "4. + Site Shortcut Trap (Full Recipe)",
            "cross_attn": True,
            "aux_labels": True,
            "trap": True,
            "description": "Full model with site bias table a[site, finding] + acquisition MLP g(params)"
        }
    ]

    ablation_records = []

    print("=" * 70)
    print("STARTING PHASE 3 ABLATION MATRIX EXPERIMENTS")
    print("=" * 70)

    for exp in experiments:
        print(f"\nRunning Experiment: {exp['name']}")
        metrics = train_one_fold(
            fold=0,
            folds_df=folds_df,
            target_cols=target_cols,
            epochs=epochs,
            device=device,
            use_cross_attention=exp["cross_attn"],
            use_aux_labels=exp["aux_labels"],
            use_trap=exp["trap"],
            output_dir=output_dir
        )
        ablation_records.append({
            "Configuration": exp["name"],
            "Description": exp["description"],
            "Validation Macro-AUC": metrics["macro_auc"],
            "Validation Log Loss": metrics["mean_log_loss"],
            "Cross-Attention": exp["cross_attn"],
            "Auxiliary Supervision": exp["aux_labels"],
            "Site Shortcut Trap": exp["trap"]
        })

    ablation_df = pd.DataFrame(ablation_records)
    csv_path = os.path.join(output_dir, "ablation_table.csv")
    json_path = os.path.join(output_dir, "ablation_table.json")
    ablation_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(ablation_records, f, indent=2)

    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS (Fold 0 Validation)")
    print("=" * 70)
    print(ablation_df[["Configuration", "Validation Macro-AUC", "Validation Log Loss"]])
    print(f"\n[+] Saved ablation table to {csv_path} and {json_path}")
    return ablation_df


if __name__ == "__main__":
    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    run_ablations(device=device)
