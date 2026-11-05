"""Text-Guided Visual Cross-Attention module.
Allows multilingual radiology report embeddings to guide visual feature representations during training.
Features:
1. Multi-head cross attention between visual queries and report token keys/values.
2. Text-modality dropout (p = 0.3 - 0.5) to train visual pathways to operate independently.
3. Learnable null token to stand in when text is absent.
4. Strict image-only pathway for test-time inference where reports are unavailable.
"""

from typing import Optional, Tuple
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

if torch is not None:
    class TextGuidedCrossAttention(nn.Module):
        """Cross-attention mechanism between visual features and radiology report text tokens."""
        def __init__(
            self,
            vis_dim: int = 512,
            text_dim: int = 768,
            num_heads: int = 8,
            dropout_p: float = 0.4
        ):
            super().__init__()
            self.vis_dim = vis_dim
            self.text_dim = text_dim
            self.num_heads = num_heads
            self.dropout_p = dropout_p
            
            # Linear projections
            self.q_proj = nn.Linear(vis_dim, vis_dim)
            self.k_proj = nn.Linear(text_dim, vis_dim)
            self.v_proj = nn.Linear(text_dim, vis_dim)
            self.out_proj = nn.Linear(vis_dim, vis_dim)
            
            self.layer_norm = nn.LayerNorm(vis_dim)
            self.drop = nn.Dropout(0.1)
            
            # Learnable null text token (used during inference or text-dropout)
            self.null_token = nn.Parameter(torch.randn(1, 1, text_dim) * 0.02)
            
        def forward(
            self,
            vis_feats: torch.Tensor,
            text_feats: Optional[torch.Tensor] = None,
            force_image_only: bool = False
        ) -> torch.Tensor:
            """Args:
                vis_feats: Tensor of shape (B, N_vis, vis_dim) or (N_vis, vis_dim)
                text_feats: Optional Tensor of shape (B, L_text, text_dim)
                force_image_only: If True, bypasses text and uses the learnable null token.
            Returns:
                Tensor of shape identical to vis_feats, enriched with text guidance if available.
            """
            is_unbatched = vis_feats.dim() == 2
            if is_unbatched:
                vis_feats = vis_feats.unsqueeze(0)  # (1, N_vis, vis_dim)
                
            batch_size = vis_feats.shape[0]
            
            # Determine whether to use text or null token
            use_null = force_image_only or (text_feats is None)
            
            if self.training and not use_null:
                # Text modality dropout during training
                if torch.rand(1).item() < self.dropout_p:
                    use_null = True
                    
            if use_null:
                # Expand learnable null token across batch
                text_inputs = self.null_token.expand(batch_size, 1, -1)
            else:
                if text_feats.dim() == 2:
                    text_inputs = text_feats.unsqueeze(0)
                else:
                    text_inputs = text_feats
                    
            # Compute Queries, Keys, Values
            q = self.q_proj(vis_feats)     # (B, N_vis, D)
            k = self.k_proj(text_inputs)  # (B, L_text, D)
            v = self.v_proj(text_inputs)  # (B, L_text, D)
            
            # Multi-head attention
            d_head = self.vis_dim // self.num_heads
            B, N_q, _ = q.shape
            _, N_k, _ = k.shape
            
            q_heads = q.view(B, N_q, self.num_heads, d_head).transpose(1, 2)  # (B, H, N_q, d_h)
            k_heads = k.view(B, N_k, self.num_heads, d_head).transpose(1, 2)  # (B, H, N_k, d_h)
            v_heads = v.view(B, N_k, self.num_heads, d_head).transpose(1, 2)  # (B, H, N_k, d_h)
            
            attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(d_head)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.drop(attn_probs)
            
            context = torch.matmul(attn_probs, v_heads)  # (B, H, N_q, d_h)
            context = context.transpose(1, 2).contiguous().view(B, N_q, self.vis_dim)
            context = self.out_proj(context)
            
            # Residual connection and layer normalization
            out = self.layer_norm(vis_feats + self.drop(context))
            
            if is_unbatched:
                out = out.squeeze(0)
                
            return out

