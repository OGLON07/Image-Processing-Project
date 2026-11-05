"""Loss functions for RSNA Knee Abnormality Detection.
Includes:
1. Weighted Multi-Label Binary Cross Entropy (BCEWithLogits) to handle class imbalance.
2. Confidence-weighted Soft BCE loss for auxiliary report-derived supervision.
3. Gaussian prior penalty on site shortcut bias table.
"""

from typing import Optional, Dict, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

if torch is not None:
    class WeightedMultiLabelBCELoss(nn.Module):
        """Weighted Multi-Label BCE loss addressing label imbalance across the 12 findings."""
        def __init__(self, pos_weights: Optional[torch.Tensor] = None):
            super().__init__()
            if pos_weights is not None:
                self.register_buffer("pos_weights", pos_weights)
            else:
                self.pos_weights = None
                
        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """Args:
                logits: Tensor of shape (..., 12)
                targets: Tensor of shape (..., 12)
            """
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weights, reduction="mean"
            )

    class ConfidenceWeightedSoftLoss(nn.Module):
        """Confidence-weighted BCE loss for soft targets derived from multilingual reports."""
        def __init__(self, eps: float = 1e-7):
            super().__init__()
            self.eps = eps
            
        def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor, confidences: torch.Tensor) -> torch.Tensor:
            """Args:
                logits: Tensor of shape (..., 12)
                soft_targets: Soft probabilities in [0, 1] of shape (..., 12)
                confidences: Confidence weights in [0, 1] of shape (..., 12)
            """
            probs = torch.sigmoid(logits)
            probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
            
            bce = -(soft_targets * torch.log(probs) + (1.0 - soft_targets) * torch.log(1.0 - probs))
            weighted_bce = bce * confidences
            return weighted_bce.mean()

    class RSNACombinedLoss(nn.Module):
        """Combined challenge loss:
        - Exact weighted BCE when ground truth image labels are available
        - Confidence-weighted BCE on soft report-derived labels when only reports exist
        - Site random effect prior regularization penalty
        """
        def __init__(self, pos_weights: Optional[torch.Tensor] = None, aux_weight: float = 0.5):
            super().__init__()
            self.bce = WeightedMultiLabelBCELoss(pos_weights=pos_weights)
            self.soft_bce = ConfidenceWeightedSoftLoss()
            self.aux_weight = aux_weight
            
        def forward(
            self,
            logits: torch.Tensor,
            gt_targets: Optional[torch.Tensor] = None,
            soft_targets: Optional[torch.Tensor] = None,
            confidences: Optional[torch.Tensor] = None,
            prior_penalty: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            loss_dict = {}
            total_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            if gt_targets is not None and not torch.isnan(gt_targets).all():
                # Ground truth image-labeled study
                valid_mask = ~torch.isnan(gt_targets)
                if valid_mask.any():
                    gt_loss = self.bce(logits[valid_mask], gt_targets[valid_mask])
                    total_loss = total_loss + gt_loss
                    loss_dict["gt_loss"] = float(gt_loss.item())
                    
            elif soft_targets is not None and confidences is not None:
                # Report-only study with auxiliary extracted labels
                aux_loss = self.soft_bce(logits, soft_targets, confidences)
                total_loss = total_loss + self.aux_weight * aux_loss
                loss_dict["aux_loss"] = float(aux_loss.item())
                
            if prior_penalty is not None:
                total_loss = total_loss + prior_penalty
                loss_dict["prior_penalty"] = float(prior_penalty.item())
                
            loss_dict["total_loss"] = float(total_loss.item())
            return total_loss, loss_dict

