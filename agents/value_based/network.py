"""Neural network architectures for value-based agents (DQN / Double DQN / Rainbow)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Maps observations to Q-values for each discrete action.

    An MLP with ReLU activations between linear layers and a linear head over actions.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (128, 64, 32),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, act_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q(s, a) for every action."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Rainbow building blocks
# ---------------------------------------------------------------------------


class NoisyLinear(nn.Module):
    """
    Factorised Gaussian noisy linear layer (Fortunato et al., 2018).

    Replaces epsilon-greedy: weights are parameterised as
        w = mu + sigma * epsilon
    where epsilon is factorised noise. The network learns mu and sigma end-to-end;
    during `training=True` forward passes noise is injected, during `eval()` the
    layer is deterministic (uses mu only).
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


class RainbowNetwork(nn.Module):
    """
    Rainbow's value network: Dueling + Noisy + Distributional (C51).

    Forward pass returns a probability distribution over `n_atoms` returns
    per action, shape [B, act_dim, n_atoms]. `q_values(x)` returns the
    expectation over atoms, shape [B, act_dim].

    Dueling decomposition on the per-atom logits:
        Q_logits(s, a) = V_logits(s) + (A_logits(s, a) - mean_a A_logits(s, a))
    Softmax is applied per (state, action) over atoms to get a valid p.m.f.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden: int = 128,
        noisy_sigma: float = 0.5,
    ) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )

        self.value_hidden = NoisyLinear(hidden, hidden, noisy_sigma)
        self.value_head = NoisyLinear(hidden, n_atoms, noisy_sigma)

        self.advantage_hidden = NoisyLinear(hidden, hidden, noisy_sigma)
        self.advantage_head = NoisyLinear(hidden, act_dim * n_atoms, noisy_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-action return distribution p(z | s, a), shape [B, A, n_atoms]."""
        f = self.feature(x)

        v = self.value_head(F.relu(self.value_hidden(f))).view(-1, 1, self.n_atoms)
        a = self.advantage_head(F.relu(self.advantage_hidden(f))).view(
            -1, self.act_dim, self.n_atoms
        )
        q_logits = v + a - a.mean(dim=1, keepdim=True)
        return F.softmax(q_logits, dim=-1)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """E[Z | s, a] = sum_z p(z | s, a) * z, shape [B, A]."""
        dist = self.forward(x)
        return (dist * self.support).sum(dim=-1)

    def reset_noise(self) -> None:
        """Resample factorised Gaussian noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
