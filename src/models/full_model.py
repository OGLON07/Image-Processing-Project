"""Full Multi-Modal Model for RSNA Knee Abnormality Detection.
Integrates:
1. 2.5D MRI Visual Backbone
2. Text-Guided Visual Cross-Attention (with learnable null token)
3. Two-Level Attention Multi-Instance Learning (Instance-to-Series + Series-to-Study)
4. Site Shortcut Debiasing Trap Branch (Training only)
5. Guaranteed Image-Only Inference Pipeline (No text inputs allowed or required)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from src.models.backbone import MRIBackbone
    from src.models.mil_pooling import TwoLevelAttentionMIL
    from src.models.cross_attention import TextGuidedCrossAttention
    from src.models.trap_branch import SiteShortcutTrap
except ImportError:
    torch = None
    nn = None
    F = None

if torch is not None:
    class RSNAKneeModel(nn.Module):
        """Complete RSNA Knee Abnormality Detection Model."""
        def __init__(
            self,
            feature_dim: int = 512,
            text_dim: int = 768,
            num_findings: int = 12,
            num_sites: int = 50,
            text_dropout_p: float = 0.4
        ):
            super().__init__()
            self.feature_dim = feature_dim
            self.num_findings = num_findings
            
            # 1. 2.5D Visual Backbone
            self.backbone = MRIBackbone(in_channels=3, feature_dim=feature_dim)
            
            # 2. Text-guided Cross Attention
            self.cross_attention = TextGuidedCrossAttention(
                vis_dim=feature_dim,
                text_dim=text_dim,
                dropout_p=text_dropout_p
            )
            
            # 3. Two-Level Attention MIL
            self.mil_pooler = TwoLevelAttentionMIL(
                in_features=feature_dim,
                hidden_dim=256,
                num_findings=num_findings
            )
            
            # 4. Training-only Site Shortcut Trap
            self.trap = SiteShortcutTrap(
                num_sites=num_sites,
                num_findings=num_findings,
                acq_dim=4
            )
            
        def forward(
            self,
            study_series_windows: List[torch.Tensor],
            text_tokens: Optional[torch.Tensor] = None,
            site_id: Optional[torch.Tensor] = None,
            acq_params: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """Multi-modal training forward pass.
            Args:
                study_series_windows: List of Tensors, one per series in study.
                                      Each Tensor has shape (N_windows, 3, H, W).
                text_tokens: Optional Tensor of shape (L_seq, text_dim) from report.
                site_id: Optional pseudo-site index Tensor () or (1,).
                acq_params: Optional acquisition parameters Tensor (S_series, 4).
            Returns:
                logits: Tensor of shape (12,)
            """
            series_feat_list = []
            
            # Process each series in the study
            for s_win in study_series_windows:
                # s_win: (N_windows, 3, H, W)
                w_vis = self.backbone(s_win)  # (N_windows, feature_dim)
                
                # Cross-attention with text if present
                w_guided = self.cross_attention(w_vis, text_feats=text_tokens)
                series_feat_list.append(w_guided)
                
            # Two-level MIL aggregation -> pure image logits
            image_logits = self.mil_pooler(series_feat_list)  # (12,)
            
            # Add training trap offset if in training mode
            if self.training:
                trap_offset = self.trap(site_id, acq_params)
                if isinstance(trap_offset, torch.Tensor):
                    total_logits = image_logits + trap_offset.squeeze(0) if trap_offset.dim() > 1 else image_logits + trap_offset
                else:
                    total_logits = image_logits
            else:
                total_logits = image_logits
                
            return total_logits
            
        @torch.no_grad()
        def forward_image_only(
            self,
            study_series_windows: List[torch.Tensor]
        ) -> torch.Tensor:
            """Guaranteed image-only inference pass for test evaluation.
            NEVER reads, requires, or references text tokens or report metadata.
            Completely isolates the visual pathway and disables the site trap branch.
            
            Args:
                study_series_windows: List of Tensors, one per series in study.
                                      Each Tensor has shape (N_windows, 3, H, W).
            Returns:
                probabilities: Tensor of shape (12,) with values in [0, 1].
            """
            self.eval()
            series_feat_list = []
            
            for s_win in study_series_windows:
                w_vis = self.backbone(s_win)  # (N_windows, feature_dim)
                
                # Force image-only cross-attention (uses learnable null token)
                w_guided = self.cross_attention(w_vis, text_feats=None, force_image_only=True)
                series_feat_list.append(w_guided)
                
            # Image-only MIL pooling
            image_logits = self.mil_pooler(series_feat_list)  # (12,)
            
            # Predict probabilities via sigmoid
            probs = torch.sigmoid(image_logits)
            return probs

