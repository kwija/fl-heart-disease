"""
Neural network architectures for heart disease classification.
"""
import torch.nn as nn

from .config import H1, H2, OUT, DROPOUT


class HeartDiseaseMLP(nn.Module):
    """
    Lightweight tabular MLP for binary classification.

    Architecture: input_dim → 64 (BN/ReLU/Drop) → 32 (BN/ReLU/Drop) → 2
    """

    def __init__(self, input_dim=13, h1=H1, h2=H2, out=OUT, drop=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h2, out),
        )

    def forward(self, x):
        return self.net(x)
