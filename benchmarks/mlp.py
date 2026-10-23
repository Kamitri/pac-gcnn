"""
MLP baseline.
"""

import torch
from torch import nn


class MLP(nn.Module):
    """
    Predict molecular properties from molecular fingerprints with a multilayer perceptron.

    Parameters
    ----------
    input_dim : int, default 2048
        Number of fingerprint bits.
    hidden_dim : int, default 512
        Number of features of the first hidden layer. The second and the third hidden layer have `hidden_dim` // 2 and
        `hidden_dim` // 4 features.
    num_tasks : int, default 1
        Number of target properties.
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 512, num_tasks: int = 1
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(  # Love how simple this model is
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
