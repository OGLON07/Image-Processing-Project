"""Exploratory Data Analysis (EDA) and Site Bias Analysis for RSNA Knee Challenge.
Phase 0 Implementation:
- CSV schemas, rows, columns, memory usage
- Label distributions, co-occurrence, and missingness
- Series distribution per study, anatomical planes, acquisition parameters
- Multilingual report statistics, language detection / site proxy
- System resources and Kaggle CLI verification
"""

import os
import re
import json
import numpy as np
import pandas as pd

def detect_language_proxy(text: str) -> str:
    """Heuristic language detection based on characteristic diagnostic terms."""
    if not isinstance(text, str) or not text.strip():
        return "Unknown"
    text_lower = text.lower()
    
    # Spanish keywords
    if any(k in text_lower for k in ["rotura", "menisco", "derrame", "líquido", "artrosis", "derecha", "izquierda", "cúndilo", "hallazgos"]):
        return "Spanish"
    # Dutch keywords
    elif any(k in text_lower for k in ["knie", "inlichtingen", "vraagstelling", "bevindingen", "scheur", "geen", "gewogen"]):
        return "Dutch"
    # French keywords
    elif any(k in text_lower for k in ["constatations", "déchirure", "épanchement", "ménisque", "antérieur", "postérieur", "genou"]):
        return "French"
    # German keywords
    elif any(k in text_lower for k in ["befund", "kreuzband", "meniskus", "gelenk", "kniegelenk", "kein"]):
        return "German"
    # Portuguese keywords
    elif any(k in text_lower for k in ["joelho", "derrame", "ligamento", "conclusão", "rotura"]):
        return "Portuguese"
    # English keywords
    elif any(k in text_lower for k in ["meniscus", "tear", "effusion", "ligament", "cartilage", "compartment", "intact", "anterior"]):
        return "English"
    else:
        return "Other"

def run_eda(data_dir: str = ".", output_dir: str = "artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("PHASE 0: EXPLORATORY DATA ANALYSIS & AUDIT")
    print("=" * 60)
    
    train_path = os.path.join(data_dir, "train.csv")
    series_path = os.path.join(data_dir, "train_series.csv")
    sub_path = os.path.join(data_dir, "sample_submission.csv")
    
    # 1. Inspect schemas and shapes
    train_df = pd.read_csv(train_path)
    series_df = pd.read_csv(series_path)
    sub_df = pd.read_csv(sub_path)
    
    print(f"train.csv shape: {train_df.shape}")
    print(f"train_series.csv shape: {series_df.shape}")
    print(f"sample_submission.csv shape: {sub_df.shape}")
    
    target_cols = [c for c in train_df.columns if c not in ["StudyInstanceUID", "Report"]]
    print(f"\n12 Abnormality Targets: {target_cols}")
    
    # 2. Missing value audit
    missing_train = train_df.isnull().sum().to_dict()
    missing_series = series_df.isnull().sum().to_dict()
    print(f"\nMissing values in train.csv:\n{json.dumps(missing_train, indent=2)}")
    
    # Image-labeled vs report-only studies
    labeled_mask = train_df[target_cols[0]].notnull()
    n_labeled = int(labeled_mask.sum())
    n_report_only = int((~labeled_mask).sum())
    print(f"\nGround Truth Image-labeled Studies: {n_labeled} ({n_labeled / len(train_df):.2%})")
    print(f"Report-Only Studies: {n_report_only} ({n_report_only / len(train_df):.2%})")
    
    # 3. Target distributions on ground-truth labeled studies
    labeled_df = train_df[labeled_mask]
    label_stats = {}
    for col in target_cols:
        pos_count = int(labeled_df[col].sum())
        pos_rate = float(labeled_df[col].mean())
        label_stats[col] = {
            "positive_count": pos_count,
            "total": n_labeled,
            "positive_rate": round(pos_rate, 4),
            "negative_count": n_labeled - pos_count,
            "pos_weight": round((n_labeled - pos_count) / max(1, pos_count), 4)
        }
    
    # 4. Correlation among findings
    corr_matrix = labeled_df[target_cols].corr().fillna(0).round(4).to_dict()
    
    # 5. Series per study and plane distribution
    series_per_study = series_df.groupby("StudyInstanceUID")["SeriesInstanceUID"].count()
    plane_dist = series_df["Anatomical_Plane"].value_counts().to_dict()
    fluid_dist = series_df["Fluid_Sensitive"].value_counts().to_dict()
    fat_dist = series_df["Fat_Suppression"].value_counts().to_dict()
    
    # 6. Report Language Analysis & Site Bias
    train_df["Detected_Language"] = train_df["Report"].apply(detect_language_proxy)
    lang_dist = train_df["Detected_Language"].value_counts().to_dict()
    
    # Language distribution among the 58 labeled vs all
    labeled_train_df = train_df[labeled_mask]
    lang_dist_labeled = labeled_train_df["Detected_Language"].value_counts().to_dict()
    
    report_lens = train_df["Report"].str.len()
    report_word_counts = train_df["Report"].str.split().str.len()
    
    report_stats = {
        "char_length_min": int(report_lens.min()),
        "char_length_median": float(report_lens.median()),
        "char_length_max": int(report_lens.max()),
        "char_length_mean": round(float(report_lens.mean()), 2),
        "word_count_median": float(report_word_counts.median()),
        "word_count_mean": round(float(report_word_counts.mean()), 2),
        "language_distribution_total": lang_dist,
        "language_distribution_labeled_58": lang_dist_labeled
    }
    
    # 7. Compile complete EDA artifact
    eda_summary = {
        "dataset_audit": {
            "train_studies_total": len(train_df),
            "train_series_total": len(series_df),
            "test_sample_sub_studies": len(sub_df),
            "labeled_studies_count": n_labeled,
            "report_only_studies_count": n_report_only,
            "target_columns": target_cols
        },
        "label_statistics": label_stats,
        "correlation_matrix": corr_matrix,
        "series_statistics": {
            "series_per_study_min": int(series_per_study.min()),
            "series_per_study_median": float(series_per_study.median()),
            "series_per_study_max": int(series_per_study.max()),
            "series_per_study_mean": round(float(series_per_study.mean()), 2),
            "anatomical_planes": plane_dist,
            "fluid_sensitive": fluid_dist,
            "fat_suppression": fat_dist
        },
        "report_statistics": report_stats
    }
    
    summary_file = os.path.join(output_dir, "eda_summary.json")
    with open(summary_file, "w") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"\n[+] Successfully saved EDA summary to: {summary_file}")
    
    # Print formatted summary table
    print("\n--- 12 ABNORMALITY LABEL DISTRIBUTION (58 Ground-Truth Studies) ---")
    label_table = pd.DataFrame.from_dict(label_stats, orient="index")
    print(label_table[["positive_count", "negative_count", "positive_rate", "pos_weight"]])
    
    print("\n--- ANATOMICAL PLANE DISTRIBUTION (24,371 Series) ---")
    for plane, count in plane_dist.items():
        print(f"  {plane:12s}: {count:6d} ({count / len(series_df):.2%})")
        
    print("\n--- REPORT LANGUAGE DISTRIBUTION (4,407 Studies) ---")
    for lang, count in lang_dist.items():
        print(f"  {lang:12s}: {count:6d} ({count / len(train_df):.2%})")
        
    return eda_summary

if __name__ == "__main__":
    run_eda()

