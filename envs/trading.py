"""Custom Gymnasium environment for single-asset trading with discrete actions."""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from envs.rewards import RewardScheme


class TradingEnv(gym.Env):
    """
    Single-asset trading environment.

    Observation: sliding window of normalized OHLCV data + current position.
    Actions:     0 = Hold, 1 = Buy, 2 = Sell
    """

    metadata: ClassVar[dict] = {"render_modes": ["human"]}

    # ----- action constants -----
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 20,
        initial_balance: float = 10_000.0,
        commission: float = 0.001,
        reward_scheme: str = "sharpe",
        render_mode: str | None = None,
    ) -> None:
        """
        Args:
            df:              DataFrame with columns ["Open", "High", "Low", "Close", "Volume"].
            window_size:     Number of past timesteps visible to the agent.
            initial_balance: Starting cash.
            commission:      Per-trade commission as a fraction (0.001 = 0.1%).
            reward_scheme:   One of "simple", "sharpe", "sortino".
            render_mode:     Optional Gymnasium render mode.
        """
        super().__init__()

        self._validate_df(df)
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.render_mode = render_mode

        self.reward_fn = RewardScheme(scheme=reward_scheme)

        # --- spaces ---
        # observation: (window_size, 5) OHLCV + 1 scalar for position
        # flattened into a single vector for MLP compatibility
        obs_dim = window_size * 5 + 1  # +1 for current position flag
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell

        # --- internal state (set in reset) ---
        self._current_step: int = 0
        self._balance: float = 0.0
        self._shares_held: int = 0
        self._entry_price: float = 0.0
        self._portfolio_values: list[float] = []
        self._returns: list[float] = []
        self._trade_log: list[dict] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._current_step = self.window_size
        self._balance = self.initial_balance
        self._shares_held = 0
        self._entry_price = 0.0
        self._portfolio_values = [self.initial_balance]
        self._returns = []
        self._trade_log = []

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        current_price = self._current_close()
        prev_portfolio = self._portfolio_value(current_price)

        # --- execute action ---
        self._execute(action, current_price)

        # --- advance time ---
        self._current_step += 1
        done = self._current_step >= len(self.df) - 1

        new_price = self._current_close()
        new_portfolio = self._portfolio_value(new_price)

        # --- track returns ---
        step_return = (
            (new_portfolio - prev_portfolio) / prev_portfolio if prev_portfolio > 0 else 0.0
        )
        self._returns.append(step_return)
        self._portfolio_values.append(new_portfolio)

        # --- reward ---
        reward = self.reward_fn.compute(self._returns)

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, done, False, info

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _execute(self, action: int, price: float) -> None:
        if action == self.BUY and self._shares_held == 0:
            cost = price * (1 + self.commission)
            max_shares = int(self._balance // cost)
            if max_shares > 0:
                self._shares_held = max_shares
                self._balance -= max_shares * cost
                self._entry_price = price
                self._log_trade("BUY", price, max_shares)

        elif action == self.SELL and self._shares_held > 0:
            revenue = self._shares_held * price * (1 - self.commission)
            self._balance += revenue
            self._log_trade("SELL", price, self._shares_held)
            self._shares_held = 0
            self._entry_price = 0.0

        # HOLD or invalid buy/sell → no-op

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        start = self._current_step - self.window_size
        end = self._current_step

        window = self.df.iloc[start:end][["Open", "High", "Low", "Close", "Volume"]].values

        # normalize OHLCV within the window (min-max per column)
        col_min = window.min(axis=0)
        col_max = window.max(axis=0)
        denom = col_max - col_min
        denom[denom == 0] = 1.0  # avoid division by zero
        window_norm = (window - col_min) / denom

        # position flag: 0.0 = flat, 1.0 = long
        position = np.array([1.0 if self._shares_held > 0 else 0.0], dtype=np.float32)

        obs = np.concatenate([window_norm.flatten(), position]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_close(self) -> float:
        return float(self.df.iloc[self._current_step]["Close"])

    def _portfolio_value(self, price: float) -> float:
        return self._balance + self._shares_held * price

    def _get_info(self) -> dict:
        price = self._current_close()
        return {
            "step": self._current_step,
            "balance": self._balance,
            "shares_held": self._shares_held,
            "portfolio_value": self._portfolio_value(price),
            "current_price": price,
            "trade_log": self._trade_log,
        }

    def _log_trade(self, side: str, price: float, qty: int) -> None:
        self._trade_log.append(
            {"step": self._current_step, "side": side, "price": price, "qty": qty}
        )

    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
