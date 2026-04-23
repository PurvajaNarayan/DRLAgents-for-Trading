"""Deep Q-Network (DQN) agent with optional Double DQN target."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent
from agents.value_based.network import QNetwork
from agents.value_based.replay import ReplayBuffer


class DQNAgent(BaseAgent):
    """
    Value-based agent for discrete action spaces.

    Implements the core DQN ingredients from Mnih et al. (2015):
      - Q-network + periodically-synced target network
      - Off-policy experience replay
      - Epsilon-greedy exploration with per-episode decay
      - TD(0) bootstrapped target

    When `config["double_dqn"]` is True, uses the Double DQN target
    (van Hasselt et al., 2016): action is selected by the online net
    and evaluated by the target net, which mitigates overestimation bias.

    The action-mask path matches the PPO agent: invalid actions are
    suppressed at both action-selection time and target-computation time.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: dict) -> None:
        super().__init__(obs_dim, act_dim, config)

        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 5e-4)
        self.batch_size = config.get("batch_size", 64)
        self.buffer_size = config.get("buffer_size", 50_000)
        self.train_start = config.get("train_start", 500)
        self.target_update_freq = config.get("target_update_freq", 500)
        self.double_dqn = config.get("double_dqn", True)
        self.max_grad_norm = config.get("max_grad_norm", 10.0)
        self.loss_type = config.get("loss", "mse")

        # Epsilon schedule (decayed once per episode via on_episode_end).
        self.epsilon = config.get("epsilon_start", 0.3)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

        hidden = config.get("hidden", (128, 64, 32))
        self.q_net = QNetwork(obs_dim, act_dim, hidden)
        self.target_net = QNetwork(obs_dim, act_dim, hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(capacity=self.buffer_size)

        self.train_step_count = 0

    # ------------------------------------------------------------------
    # BaseAgent API
    # ------------------------------------------------------------------

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        """
        Push a single 1-step transition into the replay buffer.

        Exists (instead of letting the trainer call `agent.buffer.push(...)`
        directly) so the training loop stays agent-generic and can also be
        used by RainbowAgent, whose n-step buffer handles storage differently.
        """
        self.buffer.push(obs, action, reward, next_obs, done, next_action_mask)

    def select_action(
        self,
        obs: np.ndarray,
        *,
        explore: bool = True,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Epsilon-greedy over valid actions. Greedy (argmax) when `explore=False`."""
        if explore and np.random.rand() < self.epsilon:
            if action_mask is not None:
                valid = np.flatnonzero(action_mask)
                if len(valid) == 0:
                    return 0
                return int(np.random.choice(valid))
            return int(np.random.randint(self.act_dim))

        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(obs).unsqueeze(0)).squeeze(0)
            if action_mask is not None:
                mask_t = torch.from_numpy(np.asarray(action_mask, dtype=bool))
                q = q.masked_fill(~mask_t, float("-inf"))
            return int(q.argmax().item())

    def learn(self, **kwargs) -> dict[str, float]:
        """Sample a mini-batch from replay and apply one TD update."""
        if len(self.buffer) < max(self.train_start, self.batch_size):
            return {"loss": 0.0, "q_mean": 0.0, "epsilon": self.epsilon}

        obs, actions, rewards, next_obs, dones, next_masks = self.buffer.sample(self.batch_size)

        # --- current Q(s,a) ---
        q_sa = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- TD target ---
        with torch.no_grad():
            if self.double_dqn:
                next_q_online = self.q_net(next_obs)
                if next_masks is not None:
                    next_q_online = next_q_online.masked_fill(~next_masks, float("-inf"))
                a_star = next_q_online.argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, a_star).squeeze(1)
            else:
                next_q_target = self.target_net(next_obs)
                if next_masks is not None:
                    next_q_target = next_q_target.masked_fill(~next_masks, float("-inf"))
                next_q = next_q_target.max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)

        if self.loss_type == "mse":
            loss = nn.functional.mse_loss(q_sa, target)
        else:
            loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_sa.mean().item()),
            "epsilon": self.epsilon,
        }

    def on_episode_end(self, episode: int, info: dict) -> None:
        """Decay epsilon once per episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step_count": self.train_step_count,
            },
            path / "dqn.pt",
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path / "dqn.pt", weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = float(ckpt.get("epsilon", self.epsilon_min))
        self.train_step_count = int(ckpt.get("train_step_count", 0))
