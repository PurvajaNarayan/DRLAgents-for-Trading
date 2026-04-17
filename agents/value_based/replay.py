"""Experience replay buffers for off-policy value-based agents."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

Transition = tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray | None]


class ReplayBuffer:
    """
    Fixed-size uniform replay buffer used by vanilla / Double DQN.

    Each stored transition is:
        (obs, action, reward, next_obs, done, next_action_mask)

    The next-state action mask is kept so the TD target can ignore
    invalid next-state actions (e.g. "buy while already holding").
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buf)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        self._buf.append(
            (
                np.asarray(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                bool(done),
                None if next_action_mask is None else np.asarray(next_action_mask, dtype=bool),
            )
        )

    def sample(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        """Uniform random mini-batch → stacked tensors."""
        batch = random.sample(self._buf, min(len(self._buf), batch_size))

        obs = torch.from_numpy(np.stack([b[0] for b in batch]))
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_obs = torch.from_numpy(np.stack([b[3] for b in batch]))
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        masks_raw = [b[5] for b in batch]
        if any(m is None for m in masks_raw):
            next_masks: torch.Tensor | None = None
        else:
            next_masks = torch.from_numpy(np.stack(masks_raw))

        return obs, actions, rewards, next_obs, dones, next_masks


# ---------------------------------------------------------------------------
# Rainbow: prioritized replay + n-step buffer
# ---------------------------------------------------------------------------


class _SumTree:
    """
    Binary sum-tree over `capacity` leaves. Internal nodes store the sum
    of their children, so sampling in proportion to priorities is O(log N).

    Layout: tree[0] is the root; leaves occupy tree[capacity - 1 : 2*capacity - 1].
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        while True:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def _retrieve(self, idx: int, s: float) -> int:
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> int:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        if idx != 0:
            self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Proportional prioritized experience replay (Schaul et al., 2016).

    Sampling probability P(i) ∝ |delta_i|^alpha.
    Importance-sampling weights w_i = (N · P(i))^(-beta), normalised by max.
    `update_priorities` must be called after every gradient step.
    """

    def __init__(self, capacity: int, alpha: float = 0.5, eps: float = 1e-6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.tree = _SumTree(capacity)
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.n_entries

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        data = (
            np.asarray(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
            None if next_action_mask is None else np.asarray(next_action_mask, dtype=bool),
        )
        # New experiences get the current max priority so they're sampled at least once.
        self.tree.add(self.max_priority ** self.alpha, data)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        list[int],
    ]:
        """Draw a prioritized mini-batch and IS-weights."""
        indices: list[int] = []
        batch: list = []
        priorities: list[float] = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            if data is None:
                # Fallback for partially-filled buffer: resample from a populated segment.
                idx, priority, data = self.tree.get(np.random.uniform(0, self.tree.total()))
            indices.append(idx)
            batch.append(data)
            priorities.append(priority)

        priorities_arr = np.asarray(priorities, dtype=np.float64)
        total = max(self.tree.total(), 1e-8)
        probs = priorities_arr / total
        weights_arr = (len(self) * probs + 1e-12) ** (-beta)
        weights_arr /= weights_arr.max()

        obs = torch.from_numpy(np.stack([b[0] for b in batch]))
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_obs = torch.from_numpy(np.stack([b[3] for b in batch]))
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        masks_raw = [b[5] for b in batch]
        if any(m is None for m in masks_raw):
            next_masks: torch.Tensor | None = None
        else:
            next_masks = torch.from_numpy(np.stack(masks_raw))

        weights = torch.tensor(weights_arr, dtype=torch.float32)
        return obs, actions, rewards, next_obs, dones, next_masks, weights, indices

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        """Set priority_i = (|delta_i| + eps)^alpha."""
        for idx, err in zip(indices, td_errors):
            p = (abs(float(err)) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, abs(float(err)) + self.eps)
            self.tree.update(idx, p)


class NStepBuffer:
    """
    Rolling window of length `n` used to build multi-step transitions.

    On each `push`, if the window is full (or the episode just ended) the
    oldest 1-step transition is collapsed into an n-step transition:

        (s_t, a_t, R_t^n, s_{t+n}, done_n, mask_n)
    where R_t^n = Σ_{k=0..n-1} γ^k r_{t+k} (truncated on terminal states).
    """

    def __init__(self, n: int = 3, gamma: float = 0.99) -> None:
        self.n = n
        self.gamma = gamma
        self._buf: deque[Transition] = deque(maxlen=n)

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> list[Transition]:
        """
        Append a 1-step transition; return any emitted n-step transitions.

        Normally emits 0 or 1 transition per call. On `done=True` it flushes
        every remaining partial window so no in-flight trajectory is lost.
        """
        self._buf.append(
            (
                np.asarray(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                bool(done),
                None if next_action_mask is None else np.asarray(next_action_mask, dtype=bool),
            )
        )

        emitted: list[Transition] = []

        if done:
            while len(self._buf) > 0:
                emitted.append(self._compute_nstep())
                self._buf.popleft()
        elif len(self._buf) == self.n:
            emitted.append(self._compute_nstep())
            self._buf.popleft()

        return emitted

    def _compute_nstep(self) -> Transition:
        R = 0.0
        last_idx = len(self._buf) - 1
        terminal_next_obs: np.ndarray | None = None
        terminal_done = False
        terminal_mask: np.ndarray | None = None

        for i, (_, _, r, n_obs, d, n_mask) in enumerate(self._buf):
            R += (self.gamma**i) * r
            if d:
                last_idx = i
                terminal_next_obs = n_obs
                terminal_done = True
                terminal_mask = n_mask
                break

        s0, a0, _, _, _, _ = self._buf[0]
        if terminal_done:
            next_obs_n = terminal_next_obs
            mask_n = terminal_mask
            done_n = True
        else:
            _, _, _, next_obs_n, done_n, mask_n = self._buf[last_idx]

        return (s0, a0, float(R), np.asarray(next_obs_n, dtype=np.float32), bool(done_n), mask_n)
