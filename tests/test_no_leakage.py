"""Automated Unit Test: Zero Data Leakage Verification.
Asserts that every StudyInstanceUID, its series, and any hypothetical slices
belong strictly to exactly one fold, and that train and validation sets in any fold
have ZERO intersection.
"""

import os
import unittest
import pandas as pd

class TestNoDataLeakage(unittest.TestCase):
    def setUp(self):
        self.folds_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "train_folds.csv")
        self.series_path = os.path.join(os.path.dirname(__file__), "..", "train_series.csv")
        self.assertTrue(os.path.exists(self.folds_path), f"Folds file not found at {self.folds_path}")
        self.assertTrue(os.path.exists(self.series_path), f"Series file not found at {self.series_path}")
        self.folds_df = pd.read_csv(self.folds_path)
        self.series_df = pd.read_csv(self.series_path)
        
    def test_single_fold_per_study(self):
        """Verify each StudyInstanceUID appears in exactly one fold."""
        study_fold_counts = self.folds_df.groupby("StudyInstanceUID")["fold"].nunique()
        violating_studies = study_fold_counts[study_fold_counts > 1]
        self.assertEqual(
            len(violating_studies), 0,
            f"Data leakage detected! Studies present in multiple folds: {violating_studies.index.tolist()}"
        )
        
    def test_train_val_split_disjointness(self):
        """Verify for every fold k in 0..4 that train_studies and val_studies have 0 intersection."""
        folds = sorted(self.folds_df["fold"].unique())
        self.assertEqual(len(folds), 5, f"Expected 5 folds, got {len(folds)}")
        
        for k in folds:
            val_uids = set(self.folds_df[self.folds_df["fold"] == k]["StudyInstanceUID"])
            train_uids = set(self.folds_df[self.folds_df["fold"] != k]["StudyInstanceUID"])
            
            intersection = val_uids.intersection(train_uids)
            self.assertEqual(
                len(intersection), 0,
                f"Data leakage in Fold {k}! {len(intersection)} studies overlap between train and val."
            )
            
    def test_series_level_isolation(self):
        """Verify that all series of a StudyInstanceUID inherit exactly the study's fold."""
        merged = self.series_df.merge(self.folds_df[["StudyInstanceUID", "fold"]], on="StudyInstanceUID", how="left")
        self.assertFalse(merged["fold"].isnull().any(), "Some series belong to unknown StudyInstanceUIDs!")
        
        # Check that no series for a study in val fold k appears in any train fold
        for k in sorted(self.folds_df["fold"].unique()):
            val_studies = set(self.folds_df[self.folds_df["fold"] == k]["StudyInstanceUID"])
            val_series = set(merged[merged["fold"] == k]["SeriesInstanceUID"])
            train_series = set(merged[merged["fold"] != k]["SeriesInstanceUID"])
            
            overlap = val_series.intersection(train_series)
            self.assertEqual(
                len(overlap), 0,
                f"Series leakage in Fold {k}! {len(overlap)} series appear in both train and val."
            )

if __name__ == "__main__":
    unittest.main()

