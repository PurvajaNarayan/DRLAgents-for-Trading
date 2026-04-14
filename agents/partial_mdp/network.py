"""Neural network architectures for Partial MDP agents."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """
    Predicts the next observation given the current observation and action.
    Maintains a hidden state across timesteps.
    """

    def __init__(self, obs_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        # Market dim excludes the position feature at the end
        self.market_dim = obs_dim - 1
        self.hidden_dim = hidden

        self.lstm = nn.LSTM(input_size=self.market_dim, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, self.market_dim)

    def forward(
        self,
        obs: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the predictor (market only).

        Args:
            obs: Tensor of shape (batch, seq_len, obs_dim).
            hx: Tuple of (h_0, c_0) for LSTM initial state.

        Returns:
            pred_market_obs: Tensor of shape (batch, seq_len, market_dim).
            hx_n: Tuple of (h_n, c_n) at the end of the sequence.
        """
        # Exclude the agent's position flag from the features
        market_obs = obs[..., :-1]
        
        out, hx_n = self.lstm(market_obs, hx)
        pred_market_obs = self.out(out)
        
        return pred_market_obs, hx_n


class PMDPActorCriticNetwork(nn.Module):
    """
    Actor-Critic that uses the current observation and the LSTM hidden state.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, act_dim: int, mlp_hidden: int = 128) -> None:
        super().__init__()

        in_dim = obs_dim + hidden_dim

        self.actor = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, act_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, obs: torch.Tensor, lstm_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns raw action logits and scalar value estimate."""
        x = torch.cat([obs, lstm_h], dim=-1)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def get_distribution(
        self, obs: torch.Tensor, lstm_h: torch.Tensor, action_mask: torch.Tensor | None = None
    ) -> torch.distributions.Categorical:
        """Returns a Categorical distribution over actions."""
        x = torch.cat([obs, lstm_h], dim=-1)
        logits = self.actor(x)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
            
        return torch.distributions.Categorical(logits=logits)
