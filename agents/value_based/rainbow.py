"""Rainbow DQN agent (Hessel et al., 2018).

Combines six value-based improvements over vanilla DQN:
    1. Double DQN:         action-selection / evaluation decoupled.
    2. Dueling networks:   separate value and advantage streams.
    3. Prioritized replay: sample transitions by |TD-error|.
    4. Multi-step (n-step) returns.
    5. Distributional RL (C51): predict a return distribution on a fixed
       support of `n_atoms` points in [v_min, v_max].
    6. Noisy nets:         state-dependent Gaussian noise on weights
                           replaces epsilon-greedy exploration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent
from agents.value_based.network import RainbowNetwork
from agents.value_based.replay import NStepBuffer, PrioritizedReplayBuffer


class RainbowAgent(BaseAgent):
    """
    Rainbow DQN agent for discrete action spaces.

    The training pipeline is:
        env step -> n_step_buffer.push() -> prioritized_replay.push()
        every `update_every`:
            sample(batch, beta) -> project distributional target
            -> KL loss weighted by IS-weights -> grad step
            -> update priorities with per-sample loss
        every `target_update_freq` grad steps: target_net <- q_net

    Action masking is threaded through both action-selection and next-state
    argmax (Double-DQN side) to respect `TradingEnv.get_action_mask()`.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: dict) -> None:
        super().__init__(obs_dim, act_dim, config)

        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("lr", 1e-4)
        self.batch_size = config.get("batch_size", 32)
        self.buffer_size = config.get("buffer_size", 50_000)
        self.train_start = config.get("train_start", 500)
        self.target_update_freq = config.get("target_update_freq", 500)
        self.max_grad_norm = config.get("max_grad_norm", 10.0)

        # Distributional
        self.n_atoms = config.get("n_atoms", 51)
        self.v_min = config.get("v_min", -10.0)
        self.v_max = config.get("v_max", 10.0)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        # Multi-step
        self.n_step = config.get("n_step", 3)

        # Prioritized replay
        self.per_alpha = config.get("per_alpha", 0.5)
        self.per_beta_start = config.get("per_beta_start", 0.4)
        self.per_beta_end = config.get("per_beta_end", 1.0)
        self.per_beta_steps = config.get("per_beta_steps", 100_000)

        # Noisy
        self.noisy_sigma = config.get("noisy_sigma", 0.5)

        hidden = config.get("hidden", 128)
        self.q_net = RainbowNetwork(
            obs_dim, act_dim, self.n_atoms, self.v_min, self.v_max, hidden, self.noisy_sigma
        )
        self.target_net = RainbowNetwork(
            obs_dim, act_dim, self.n_atoms, self.v_min, self.v_max, hidden, self.noisy_sigma
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr, eps=1.5e-4)
        self.buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.per_alpha)
        self.n_step_buffer = NStepBuffer(n=self.n_step, gamma=self.gamma)

        self.train_step_count = 0

        # Trainer compatibility: existing log lines reference `agent.epsilon`.
        # Rainbow uses noisy nets for exploration, so expose a constant 0.0.
        self.epsilon = 0.0

    # ------------------------------------------------------------------
    # Rollout-side API (matches DQNAgent.store_transition)
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
        """Feed a 1-step transition through the n-step window into the PER buffer."""
        for nstep_trans in self.n_step_buffer.push(
            obs, action, reward, next_obs, done, next_action_mask
        ):
            s0, a0, R, sn, d_n, m_n = nstep_trans
            self.buffer.push(s0, a0, R, sn, d_n, m_n)

    def select_action(
        self,
        obs: np.ndarray,
        *,
        explore: bool = True,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """
        Rainbow does not use epsilon-greedy. Exploration comes from the
        factorised Gaussian noise inside the online network's NoisyLinear
        layers (live while `.training=True`). At eval time the network is
        put in `.eval()` so only the mean weights are used.
        """
        was_training = self.q_net.training
        if explore:
            self.q_net.train()
            self.q_net.reset_noise()
        else:
            self.q_net.eval()

        try:
            with torch.no_grad():
                q = self.q_net.q_values(torch.FloatTensor(obs).unsqueeze(0)).squeeze(0)
                if action_mask is not None:
                    mask_t = torch.from_numpy(np.asarray(action_mask, dtype=bool))
                    q = q.masked_fill(~mask_t, float("-inf"))
                return int(q.argmax().item())
        finally:
            self.q_net.train(was_training)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _current_beta(self) -> float:
        frac = min(1.0, self.train_step_count / max(1, self.per_beta_steps))
        return self.per_beta_start + frac * (self.per_beta_end - self.per_beta_start)

    def learn(self, **kwargs) -> dict[str, float]:
        if len(self.buffer) < max(self.train_start, self.batch_size):
            return {"loss": 0.0, "beta": self._current_beta(), "q_mean": 0.0, "epsilon": 0.0}

        beta = self._current_beta()
        (
            obs,
            actions,
            rewards,
            next_obs,
            dones,
            next_masks,
            weights,
            indices,
        ) = self.buffer.sample(self.batch_size, beta=beta)

        # Fresh noise on both networks for this update.
        self.q_net.train()
        self.q_net.reset_noise()
        self.target_net.reset_noise()

        B = obs.size(0)

        # --- current distribution log p(z | s, a) ---
        dist = self.q_net(obs)  # [B, A, n_atoms]
        log_p = torch.log(dist.clamp(min=1e-8))[torch.arange(B), actions]  # [B, n_atoms]

        # --- target distribution via Double DQN action selection ---
        with torch.no_grad():
            next_q_online = self.q_net.q_values(next_obs)  # [B, A]
            if next_masks is not None:
                next_q_online = next_q_online.masked_fill(~next_masks, float("-inf"))
            next_actions = next_q_online.argmax(dim=1)  # [B]

            next_dist = self.target_net(next_obs)[torch.arange(B), next_actions]  # [B, n_atoms]

            support = self.q_net.support  # [n_atoms]
            gamma_n = self.gamma**self.n_step
            Tz = rewards.unsqueeze(1) + gamma_n * support.unsqueeze(0) * (1.0 - dones.unsqueeze(1))
            Tz = Tz.clamp(self.v_min, self.v_max)

            b = (Tz - self.v_min) / self.delta_z  # [B, n_atoms]
            lower = b.floor().long()
            upper = b.ceil().long()
            lower = lower.clamp(0, self.n_atoms - 1)
            upper = upper.clamp(0, self.n_atoms - 1)

            # Handle the edge case b == integer exactly -> lower == upper.
            same = (lower == upper).float()
            m = torch.zeros_like(next_dist)
            offset = (
                torch.arange(B, dtype=torch.long).unsqueeze(1).expand(B, self.n_atoms)
                * self.n_atoms
            )

            m.view(-1).index_add_(
                0,
                (lower + offset).view(-1),
                (next_dist * ((upper.float() - b) + same)).view(-1),
            )
            m.view(-1).index_add_(
                0,
                (upper + offset).view(-1),
                (next_dist * (b - lower.float())).view(-1),
            )

        per_sample_loss = -(m * log_p).sum(dim=1)  # [B]
        loss = (per_sample_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.buffer.update_priorities(indices, per_sample_loss.detach().cpu().numpy())

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        with torch.no_grad():
            q_mean = float(self.q_net.q_values(obs).mean().item())

        return {
            "loss": float(loss.item()),
            "beta": float(beta),
            "q_mean": q_mean,
            "epsilon": 0.0,
        }

    def on_episode_end(self, episode: int, info: dict) -> None:
        """Clear n-step window so partial transitions don't leak across episodes."""
        self.n_step_buffer.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_step_count": self.train_step_count,
                "n_atoms": self.n_atoms,
                "v_min": self.v_min,
                "v_max": self.v_max,
                "n_step": self.n_step,
            },
            path / "rainbow.pt",
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path / "rainbow.pt", weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_step_count = int(ckpt.get("train_step_count", 0))
