"""Site Shortcut Trap Module (Training-Only Debiasing Branch).
Implements the 33rd Place Silver Solution's key innovation:
Training:
    logit = image_logit + a[site, finding] + mean_{series}(g(acquisition_params)[finding])
Inference:
    logit = image_logit

Components:
1. a[site, finding]: Learnable site random-effect table regularized toward 0 with N(0, 0.5^2) prior,
   initialized with empirical-Bayes shrunk site log-odds.
2. g(acquisition_params): Lightweight MLP on series acquisition features (plane, fluid-sensitive, fat-suppressed, slice count)
   with zero-initialized final layer.
3. image-only pass: Strict bypass during inference so that site reporting bias cannot leak into final predictions.
"""

from typing import Optional, List, Dict
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
    class SiteShortcutTrap(nn.Module):
        """Training-only debiasing trap branch."""
        def __init__(
            self,
            num_sites: int = 50,
            num_findings: int = 12,
            acq_dim: int = 4,
            hidden_dim: int = 32
        ):
            super().__init__()
            self.num_sites = num_sites
            self.num_findings = num_findings
            
            # Site random effect embedding table: (num_sites, num_findings)
            self.site_bias = nn.Embedding(num_sites, num_findings)
            # Initialize with small variance
            nn.init.normal_(self.site_bias.weight, mean=0.0, std=0.1)
            
            # Acquisition parameters MLP g: (acq_dim -> hidden_dim -> num_findings)
            self.acq_mlp = nn.Sequential(
                nn.Linear(acq_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_findings)
            )
            # Zero-initialize the final layer so training starts from pure image logits
            nn.init.zeros_(self.acq_mlp[-1].weight)
            nn.init.zeros_(self.acq_mlp[-1].bias)
            
        def init_empirical_bayes_bias(self, site_log_odds: np.ndarray):
            """Initializes site bias embeddings from empirical-Bayes shrunk log-odds."""
            with torch.no_grad():
                tensor_init = torch.from_numpy(site_log_odds).float()
                self.site_bias.weight.data.copy_(tensor_init)
                
        def forward(
            self,
            site_idx: Optional[torch.Tensor] = None,
            acq_params: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """Computes shortcut trap adjustment logits.
            Args:
                site_idx: Tensor of shape (B,) or () containing pseudo-site IDs.
                acq_params: Tensor of shape (B, S, acq_dim) or (S, acq_dim) containing acquisition parameters.
            Returns:
                trap_logits: Tensor of shape (B, num_findings) or (num_findings,)
            """
            is_unbatched = False
            if site_idx is not None and site_idx.dim() == 0:
                site_idx = site_idx.unsqueeze(0)
                is_unbatched = True
                
            trap_offset = 0.0
            
            # 1. Site bias term a[site, finding]
            if site_idx is not None:
                site_term = self.site_bias(site_idx)  # (B, 12)
                trap_offset = trap_offset + site_term
                
            # 2. Acquisition parameter term mean_{series}(g(acq_params))
            if acq_params is not None:
                if acq_params.dim() == 2:  # (S, acq_dim)
                    acq_term = self.acq_mlp(acq_params).mean(dim=0, keepdim=True)  # (1, 12)
                else:  # (B, S, acq_dim)
                    acq_term = self.acq_mlp(acq_params).mean(dim=1)  # (B, 12)
                trap_offset = trap_offset + acq_term
                
            if is_unbatched and isinstance(trap_offset, torch.Tensor):
                trap_offset = trap_offset.squeeze(0)
                
            return trap_offset
            
        def get_prior_loss(self) -> torch.Tensor:
            """Computes Gaussian prior penalty on site biases: N(0, 0.5^2) => weight_decay = 1 / (2 * 0.25) = 2.0."""
            return 2.0 * torch.mean(self.site_bias.weight ** 2)

