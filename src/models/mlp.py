"""A multilayer perceptron on raw pixels.

TASK 2. Build `MLP`: a fully connected network that maps one flattened,
standardised photograph (`in_dim` grey levels) to one number in [0, 1].

* `hidden` is a list of layer widths from the config, for example [256, 64];
  build one Linear + nonlinearity + Dropout block per entry.
* `dropout` is the probability from the config. Where it goes, and whether
  it belongs after the last hidden layer, is your call.
* The output must lie in [0, 1]. `prior` is the mean target of the training
  rows; it is passed to you for a reason you have to find yourself.
* `forward` returns shape (batch,), not (batch, 1).

It trains in seconds on a CPU. That is the point of starting here: every
decision in it can be tested in the time it takes to read this docstring.
Count its parameters against the number of training photographs before you
decide how much dropout it needs.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        dropout: float = 0.3,
        prior: float = 0.5,
    ):
        super().__init__()

        if not 0.0 <= prior <= 1.0:
            raise ValueError("prior must be between 0 and 1.")

        layers = []
        prev_dim = in_dim

        for width in hidden:
            layers.extend([
                nn.Linear(prev_dim, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = width

        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

        # Initialize output bias so the initial prediction reflects prior.
        eps = 1e-7
        p = min(max(prior, eps), 1.0 - eps)
        logit_prior = math.log(p / (1.0 - p))

        nn.init.constant_(self.output.bias, logit_prior)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x.squeeze(-1)
