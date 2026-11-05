"""Two-Level Attention Multi-Instance Learning (MIL) module.
Level 1: Aggregates instance (slice/window) feature embeddings into series representations
         with 12 finding-specific attention heads.
Level 2: Aggregates series representations across all anatomical planes (Sagittal, Coronal, Axial)
         into study-level logits for the 12 abnormalities.
Supports variable numbers of slices and series (3-14 series per study) without padding or truncation.
"""

from typing import List, Tuple, Optional
import math

# We structure the module so it works with PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

if torch is not None:
    class SeriesInstanceAttention(nn.Module):
        """Level 1 MIL: Attention over windows/slices within a single MRI series.
        Produces 12 finding-specific series representations of dimension D.
        """
        def __init__(self, in_features: int = 512, hidden_dim: int = 256, num_findings: int = 12):
            super().__init__()
            self.in_features = in_features
            self.num_findings = num_findings
            
            # Attention network per finding (1024/512 -> 256 -> tanh -> 12)
            self.attention_net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_findings)
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Args:
                x: Tensor of shape (N_windows, in_features) representing all windows in a series.
            Returns:
                Tensor of shape (num_findings, in_features) containing finding-specific series vectors.
            """
            # Compute raw attention scores: (N_windows, num_findings)
            scores = self.attention_net(x)
            
            # Softmax over windows for each finding independently
            # Shape: (num_findings, N_windows)
            attn_weights = F.softmax(scores.transpose(0, 1), dim=-1)
            
            # Weighted sum over windows: (num_findings, in_features)
            series_vectors = torch.matmul(attn_weights, x)
            return series_vectors

    class StudySeriesAttention(nn.Module):
        """Level 2 MIL: Attention across all series in a study per finding.
        Aggregates variable number of series (3 to 14) into study-level abnormality logits.
        """
        def __init__(self, in_features: int = 512, hidden_dim: int = 128, num_findings: int = 12):
            super().__init__()
            self.num_findings = num_findings
            self.in_features = in_features
            
            # Per-finding series attention scoring
            self.series_attn_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_features, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1)
                ) for _ in range(num_findings)
            ])
            
            # Final classifier heads per finding
            self.classifiers = nn.ModuleList([
                nn.Linear(in_features, 1) for _ in range(num_findings)
            ])
            
        def forward(self, series_representations: List[torch.Tensor]) -> torch.Tensor:
            """Args:
                series_representations: List of length S (number of series in the study),
                                        where each element is a Tensor of shape (num_findings, in_features).
            Returns:
                logits: Tensor of shape (num_findings,) containing the unnormalized log-odds.
            """
            num_series = len(series_representations)
            logits = []
            
            for f_idx in range(self.num_findings):
                # Stack representations for finding f_idx across all S series: (S, in_features)
                f_series_stack = torch.stack([s_rep[f_idx] for s_rep in series_representations], dim=0)
                
                # Compute attention scores over series: (S, 1) -> (1, S)
                scores = self.series_attn_heads[f_idx](f_series_stack).squeeze(-1)
                attn_weights = F.softmax(scores, dim=0)  # (S,)
                
                # Weighted aggregation across series: (in_features,)
                study_finding_vector = torch.sum(attn_weights.unsqueeze(-1) * f_series_stack, dim=0)
                
                # Logit for this finding
                logit = self.classifiers[f_idx](study_finding_vector)
                logits.append(logit)
                
            return torch.cat(logits, dim=-1)  # Shape: (12,)

    class TwoLevelAttentionMIL(nn.Module):
        """Full Two-Level Attention MIL aggregator for MRI studies."""
        def __init__(self, in_features: int = 512, hidden_dim: int = 256, num_findings: int = 12):
            super().__init__()
            self.level1_instance_attn = SeriesInstanceAttention(in_features, hidden_dim, num_findings)
            self.level2_series_attn = StudySeriesAttention(in_features, hidden_dim // 2, num_findings)
            
        def forward(self, study_series_windows: List[torch.Tensor]) -> torch.Tensor:
            """Args:
                study_series_windows: List of Tensors, one per series in the study.
                                      Each Tensor has shape (N_windows, in_features).
            Returns:
                logits: Tensor of shape (num_findings,) = (12,)
            """
            series_vectors = []
            for w_feats in study_series_windows:
                # Level 1 pooling over windows of this series -> (12, in_features)
                s_vec = self.level1_instance_attn(w_feats)
                series_vectors.append(s_vec)
                
            # Level 2 pooling over all series of the study -> (12,)
            logits = self.level2_series_attn(series_vectors)
            return logits

