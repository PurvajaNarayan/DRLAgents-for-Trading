"""Proximal Policy Optimization (PPO) with clipped objective."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent
from agents.policy_gradient.network import PolicyNetwork, ValueNetwork


class RolloutBuffer:
    """Stores a batch of transitions collected during rollout."""

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.action_masks: list[np.ndarray | None] = []

    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        action_mask: np.ndarray | None = None,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        return len(self.obs)


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (Clip variant).

    Key idea: limit how much the policy can change per update using
    a clipped surrogate objective. This prevents destructive large updates
    that can collapse the policy.

    Loss = -min(r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t)
    where r_t = pi_new(a|s) / pi_old(a|s)
    """

    def __init__(self, obs_dim: int, act_dim: int, config: dict) -> None:
        super().__init__(obs_dim, act_dim, config)

        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.n_epochs = config.get("n_epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.entropy_coef = config.get("entropy_coef", 0.05)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.lr = config.get("lr", 3e-4)

        hidden = config.get("hidden", 128)
        self.policy = PolicyNetwork(obs_dim, act_dim, hidden)
        self.value = ValueNetwork(obs_dim, hidden)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=self.lr,
        )

        self.buffer = RolloutBuffer()

    def select_action(
        self, obs: np.ndarray, *, explore: bool = True, action_mask: np.ndarray | None = None
    ) -> int:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mask_t = torch.BoolTensor(action_mask).unsqueeze(0) if action_mask is not None else None

        with torch.no_grad():
            dist = self.policy.get_distribution(obs_t, mask_t)
            v = self.value(obs_t)

        action = dist.sample() if explore else dist.probs.argmax(dim=-1)

        log_prob = dist.log_prob(action).item()
        value = v.item()

        return action.item(), log_prob, value

    def learn(self, **kwargs) -> dict[str, float]:
        """
        Run multiple epochs of minibatch updates on the collected rollout.

        Call this after filling the buffer with a full rollout.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # --- compute GAE advantages ---
        advantages, returns = self._compute_gae()

        # --- convert buffer to tensors ---
        obs_t = torch.FloatTensor(np.array(self.buffer.obs))
        actions_t = torch.LongTensor(self.buffer.actions)
        old_log_probs_t = torch.FloatTensor(self.buffer.log_probs)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        # build mask tensor (None entries become all-True)
        masks_list = []
        for m in self.buffer.action_masks:
            if m is not None:
                masks_list.append(m)
            else:
                masks_list.append(np.ones(self.act_dim, dtype=bool))
        masks_t = torch.BoolTensor(np.array(masks_list))

        # normalize advantages
        if advantages_t.std() > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / advantages_t.std()

        # --- minibatch updates ---
        n = len(self.buffer)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n)

            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_obs = obs_t[idx]
                batch_actions = actions_t[idx]
                batch_old_lp = old_log_probs_t[idx]
                batch_adv = advantages_t[idx]
                batch_returns = returns_t[idx]
                batch_masks = masks_t[idx]

                # --- evaluate current policy on batch ---
                dist = self.policy.get_distribution(batch_obs, batch_masks)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                new_values = self.value(batch_obs)

                # --- clipped surrogate loss ---
                ratio = (new_log_probs - batch_old_lp).exp()
                surr1 = ratio * batch_adv
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- value loss ---
                value_loss = nn.functional.mse_loss(new_values, batch_returns)

                # --- combined loss ---
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def _compute_gae(self) -> tuple[list[float], list[float]]:
        """
        Generalized Advantage Estimation.

        GAE(lambda) balances bias vs variance in advantage estimates:
            A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        lambda=0 → pure TD (low variance, high bias)
        lambda=1 → pure MC (high variance, low bias)
        lambda=0.95 → good default balance
        """
        advantages = []
        returns = []
        gae = 0.0

        values = [*self.buffer.values, 0.0]  # bootstrap with 0 at terminal

        for t in reversed(range(len(self.buffer))):
            next_non_terminal = 0.0 if self.buffer.dones[t] else 1.0
            delta = (
                self.buffer.rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path / "ppo.pt",
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path / "ppo.pt", weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
