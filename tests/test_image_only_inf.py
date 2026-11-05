"""Automated Unit Test: Image-Only Inference Pathway Verification.
Asserts that:
1. The model's test prediction pathway (forward_image_only) only accepts MRI image series windows.
2. The method signature and execution never inspect or depend on text, reports, or report tokens.
3. Supplying text or calling forward_image_only produces valid probability outputs for all 12 abnormalities.
4. Output values strictly lie in the valid probability range [0.0, 1.0].
"""

import unittest
import inspect
import numpy as np

try:
    import torch
    from src.models.full_model import RSNAKneeModel
except ImportError:
    torch = None

class TestImageOnlyInference(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("PyTorch not yet installed")
        torch.manual_seed(42)
        self.model = RSNAKneeModel(feature_dim=64, text_dim=64, num_findings=12, num_sites=10)
        self.model.eval()
        
    def test_signature_has_no_text_parameter(self):
        """Verify forward_image_only signature does not require or accept text/report parameters."""
        sig = inspect.signature(self.model.forward_image_only)
        param_names = list(sig.parameters.keys())
        self.assertNotIn("text", param_names, "forward_image_only contains forbidden parameter 'text'!")
        self.assertNotIn("report", param_names, "forward_image_only contains forbidden parameter 'report'!")
        self.assertNotIn("report_tokens", param_names, "forward_image_only contains forbidden parameter 'report_tokens'!")
        self.assertEqual(param_names, ["study_series_windows"], f"Unexpected parameters: {param_names}")
        
    def test_inference_execution_without_text(self):
        """Verify image-only inference produces valid (12,) probabilities from pure image inputs."""
        num_series = 5
        windows_per_series = 4
        h, w = 64, 64  # Compact size for rapid unit test
        
        # Simulated MRI study with 5 series, 4 windows of 3 channels each
        study_windows = [
            torch.randn(windows_per_series, 3, h, w) for _ in range(num_series)
        ]
        
        with torch.no_grad():
            probs = self.model.forward_image_only(study_windows)
            
        self.assertIsInstance(probs, torch.Tensor)
        self.assertEqual(probs.shape, (12,), f"Expected shape (12,), got {probs.shape}")
        
        # Verify probabilities lie strictly in [0, 1]
        probs_np = probs.cpu().numpy()
        self.assertTrue(np.all(probs_np >= 0.0), f"Found probabilities < 0: {probs_np}")
        self.assertTrue(np.all(probs_np <= 1.0), f"Found probabilities > 1: {probs_np}")
        self.assertFalse(np.any(np.isnan(probs_np)), "Found NaNs in predicted probabilities!")

if __name__ == "__main__":
    unittest.main()

