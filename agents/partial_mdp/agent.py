"""Partial MDP Agent that coordinates LSTM Predictor and PPO Actor-Critic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent
from agents.partial_mdp.network import LSTMPredictor, PMDPActorCriticNetwork


class PMDPRolloutBuffer:
    """Stores a batch of transitions collected during rollout including LSTM hidden states."""

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.hxs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []
        self.action_masks: list[np.ndarray | None] = []

    def store(
        self,
        obs: np.ndarray,
        hx: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        action_mask: np.ndarray | None = None,
    ) -> None:
        self.obs.append(obs)
        self.hxs.append(hx)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def clear(self) -> None:
        self.obs.clear()
        self.hxs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()

    def __len__(self) -> int:
        return len(self.obs)


class PMDPAgent(BaseAgent):
    """
    Partial MDP Agent with PPO.
    Uses a pretrained, frozen LSTM to predict next states and provide context.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: dict) -> None:
        super().__init__(obs_dim, act_dim, config)

        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.n_epochs = config.get("n_epochs", 10)
        self.batch_size = config.get("batch_size", 64)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.lr_actor_critic = config.get("lr", 3e-4)
        self.lr_lstm = config.get("lr_lstm", 1e-3)

        self.hidden_dim = config.get("hidden", 128)

        # Networks
        self.predictor = LSTMPredictor(obs_dim, self.hidden_dim)
        self.actor_critic = PMDPActorCriticNetwork(obs_dim, self.hidden_dim, act_dim, 128)

        # Optimizers
        self.predictor_optim = optim.Adam(self.predictor.parameters(), lr=self.lr_lstm)
        self.ac_optim = optim.Adam(
            list(self.actor_critic.parameters()),
            lr=self.lr_actor_critic,
        )

        self.buffer = PMDPRolloutBuffer()

        # State tracking for inference/rollouts
        self._current_hx: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_hidden_state(self) -> None:
        """Called at the start of a new episode to reset LSTM context."""
        self._current_hx = None

    def update_hidden_state(self, obs: np.ndarray) -> None:
        """Steps the frozen LSTM forward one frame to update internal context."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).view(1, 1, -1)
            _, self._current_hx = self.predictor(obs_t, self._current_hx)

    @property
    def current_h(self) -> np.ndarray:
        """Returns the current h_t from the LSTM cell (or zeros if None)."""
        if self._current_hx is None:
            return np.zeros(self.hidden_dim, dtype=np.float32)
        # hx is (h, c), h is shape (num_layers=1, batch=1, hidden_size)
        return self._current_hx[0].squeeze().cpu().numpy()

    def select_action(
        self, obs: np.ndarray, *, explore: bool = True, action_mask: np.ndarray | None = None
    ) -> tuple[int, float, float]:
        """
        Samples an action using the current observation and the temporal context.
        Returns (action, log_prob, value).
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mask_t = torch.BoolTensor(action_mask).unsqueeze(0) if action_mask is not None else None
        
        # We use current_h which is based on history up to (o_{t-1}, a_{t-1})
        hx_t = torch.FloatTensor(self.current_h).unsqueeze(0)

        with torch.no_grad():
            dist = self.actor_critic.get_distribution(obs_t, hx_t, mask_t)
            _, v = self.actor_critic(obs_t, hx_t)

        action = dist.sample() if explore else dist.probs.argmax(dim=-1)

        log_prob = dist.log_prob(action).item()
        value = v.item()

        return action.item(), log_prob, value

    def freeze_predictor(self) -> None:
        """Freezes the LSTM weights after pretraining phase is complete."""
        for param in self.predictor.parameters():
            param.requires_grad = False
        self.predictor.eval()

    def learn_predictor(self, obs_list: list[np.ndarray]) -> float:
        """
        Runs one step of BPTT supervised learning on a single episodic trajectory.
        
        Args:
            obs_list: length T+1
        Returns:
            mse_loss: float
        """
        self.predictor.train()
        
        # Prepare sequences
        # Batch size = 1, Seq length = T
        obs_seq = torch.FloatTensor(np.array(obs_list[:-1])).unsqueeze(0)
        
        # Target excludes the position scalar (which is the last feature)
        target_market_seq = torch.FloatTensor(np.array(obs_list[1:])).unsqueeze(0)[..., :-1]
        
        pred_market_obs, _ = self.predictor(obs_seq)
        
        loss = nn.functional.mse_loss(pred_market_obs, target_market_seq)
        
        self.predictor_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), self.max_grad_norm)
        self.predictor_optim.step()
        
        return loss.item()

    def learn(self, **kwargs) -> dict[str, float]:
        """
        Actor-Critic structured identical to standard PPO mini-batch updates.
        Relies on the frozen LSTM features stored in `self.buffer.hxs`.
        """
        if len(self.buffer) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # --- compute GAE advantages ---
        advantages, returns = self._compute_gae()

        # --- convert buffer to tensors ---
        obs_t = torch.FloatTensor(np.array(self.buffer.obs))
        hxs_t = torch.FloatTensor(np.array(self.buffer.hxs))
        actions_t = torch.LongTensor(self.buffer.actions)
        old_log_probs_t = torch.FloatTensor(self.buffer.log_probs)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        # build mask tensor
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
                batch_hxs = hxs_t[idx]
                batch_actions = actions_t[idx]
                batch_old_lp = old_log_probs_t[idx]
                batch_adv = advantages_t[idx]
                batch_returns = returns_t[idx]
                batch_masks = masks_t[idx]

                # --- evaluate current policy on batch ---
                dist = self.actor_critic.get_distribution(batch_obs, batch_hxs, batch_masks)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                _, new_values = self.actor_critic(batch_obs, batch_hxs)

                # --- clipped surrogate loss ---
                ratio = (new_log_probs - batch_old_lp).exp()
                surr1 = ratio * batch_adv
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- value loss ---
                value_loss = nn.functional.mse_loss(new_values, batch_returns)

                # --- combined loss ---
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.ac_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.ac_optim.step()

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
        advantages = []
        returns = []
        gae = 0.0

        values = [*self.buffer.values, 0.0]

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
                "predictor": self.predictor.state_dict(),
                "actor_critic": self.actor_critic.state_dict(),
                "predictor_optim": self.predictor_optim.state_dict(),
                "ac_optim": self.ac_optim.state_dict(),
            },
            path / "pmdp.pt",
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path / "pmdp.pt", weights_only=True)
        self.predictor.load_state_dict(ckpt["predictor"])
        self.actor_critic.load_state_dict(ckpt["actor_critic"])
        self.predictor_optim.load_state_dict(ckpt["predictor_optim"])
        self.ac_optim.load_state_dict(ckpt["ac_optim"])
