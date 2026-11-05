"""Automated Unit Test: Model Pipeline, Dynamic Series Bags, and Output Shapes.
Asserts that:
1. Dynamic series bag sizes (3 series, 5 series, 14 series per study) pass through MIL without error.
2. 2.5D input channels (3 slices) are accepted.
3. Output logits and probabilities are strictly of shape (12,).
"""

import unittest
import numpy as np

try:
    import torch
    from src.models.full_model import RSNAKneeModel
except ImportError:
    torch = None

class TestModelPipeline(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("PyTorch not yet installed")
        torch.manual_seed(42)
        self.model = RSNAKneeModel(feature_dim=64, text_dim=64, num_findings=12, num_sites=10)
        
    def test_dynamic_series_counts(self):
        """Test with min (3), median (5), and max (14) series per study from train_series.csv."""
        for num_series in [3, 5, 14]:
            study_windows = [
                torch.randn(3, 3, 64, 64) for _ in range(num_series)
            ]
            self.model.eval()
            probs = self.model.forward_image_only(study_windows)
            self.assertEqual(probs.shape, (12,), f"Failed for {num_series} series")
            
    def test_multimodal_training_forward(self):
        """Test multi-modal training forward pass with text tokens and site ID."""
        self.model.train()
        study_windows = [torch.randn(2, 3, 64, 64) for _ in range(4)]
        text_tokens = torch.randn(15, 64)  # 15 tokens of text embeddings
        site_id = torch.tensor([2])
        acq_params = torch.randn(4, 4)
        
        logits = self.model(study_windows, text_tokens=text_tokens, site_id=site_id, acq_params=acq_params)
        self.assertEqual(logits.shape, (12,))
        self.assertFalse(torch.isnan(logits).any())

if __name__ == "__main__":
    unittest.main()

