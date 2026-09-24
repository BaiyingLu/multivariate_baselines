"""
Point-wise-token Transformer encoder + MLP forecast head
(Baseline A: C = 1, bg only; Baseline B: C = 3, bg + carbs + bolus).

    x (B, T_in, C)
      → Linear(C, D)                      one token per 5-min step; channels mixed here
      → + learnable positional embedding  (T_in, D)
      → N × TransformerEncoderLayer        self-attention, no causal mask (all inputs are past)
      → last token H[:, -1]               (B, D)
      → MLP: Linear(D, D/2) → GELU → Dropout → Linear(D/2, T_out)
      → ŷ (B, T_out)                      direct multi-step output
"""

import torch
import torch.nn as nn


class TransformerForecaster(nn.Module):
    def __init__(self, c_in: int, t_in: int, t_out: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, ffn_mult: int = 4, dropout: float = 0.1, norm_first: bool = True):
        super().__init__()
        self.input_proj = nn.Linear(c_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, t_in, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_mult * d_model, dropout,
            activation="gelu", batch_first=True, norm_first=norm_first,
        )
        # Pre-LN stacks need a final LayerNorm on the encoder output.
        self.encoder = nn.TransformerEncoder(
            layer, n_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, t_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.input_proj(x) + self.pos_emb)
        return self.head(h[:, -1])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
