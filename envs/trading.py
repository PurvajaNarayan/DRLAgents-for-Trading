"""Custom Gymnasium environment for single-asset trading with discrete actions."""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from envs.rewards import RewardScheme
from features.raw_ohlcv import RawOHLCV


class TradingEnv(gym.Env):
    """
    Single-asset trading environment.

    Observation: built by a pluggable feature builder (RawOHLCV or OHLCVWithIndicators).
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
        feature_builder=None,
        window_size: int = 30,
        initial_balance: float = 10_000.0,
        commission: float = 0.001,
        reward_scheme: str = "sharpe",
        render_mode: str | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        """
        Args:
            df:              DataFrame with columns ["Open", "High", "Low", "Close", "Volume"].
            feature_builder: State builder instance (RawOHLCV or OHLCVWithIndicators).
                             Defaults to RawOHLCV if not provided.
            window_size:     Number of past timesteps visible to the agent.
            initial_balance: Starting cash.
            commission:      Per-trade commission as a fraction (0.001 = 0.1%).
            reward_scheme:   One of "simple", "sharpe", "sortino".
            render_mode:     Optional Gymnasium render mode.
        """
        super().__init__()

        self._validate_df(df)

        # --- feature builder ---
        self.feature_builder = feature_builder or RawOHLCV(window_size=window_size)
        self.window_size = window_size

        # if the builder has a precompute step (e.g. technical indicators), run it
        if hasattr(self.feature_builder, "precompute"):
            self.df = self.feature_builder.precompute(df).reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.initial_balance = initial_balance
        self.commission = commission
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.reward_fn = RewardScheme(scheme=reward_scheme)

        # --- spaces ---
        obs_dim = self.feature_builder.obs_dim
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
        self._action_returns: list[float] = []
        self._trade_log: list[dict] = []
        self._steps_in_episode: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.max_episode_steps is not None:
            latest_start = max(self.window_size, len(self.df) - 1 - self.max_episode_steps)
            if latest_start == self.window_size:
                self._current_step = self.window_size
            else:
                # Use gymnasium's np_random initialized by super().reset()
                self._current_step = self.np_random.integers(self.window_size, latest_start + 1)
        else:
            self._current_step = self.window_size

        self._balance = self.initial_balance
        self._shares_held = 0
        self._entry_price = 0.0
        self._portfolio_values = [self.initial_balance]
        self._returns = []
        self._action_returns = []
        self._trade_log = []
        self._steps_in_episode = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        current_price = self._current_close()
        prev_portfolio = self._portfolio_value(current_price)

        was_holding = (self._shares_held > 0)
        entry_price_before = self._entry_price

        # --- execute action ---
        self._execute(action, current_price)

        # --- advance time ---
        self._current_step += 1
        self._steps_in_episode += 1
        
        done_eof = self._current_step >= len(self.df) - 1
        done_max_steps = (self.max_episode_steps is not None) and (self._steps_in_episode >= self.max_episode_steps)
        done = bool(done_eof or done_max_steps)

        new_price = self._current_close()
        new_portfolio = self._portfolio_value(new_price)

        # --- track returns ---
        step_return = (
            (new_portfolio - prev_portfolio) / prev_portfolio if prev_portfolio > 0 else 0.0
        )
        self._returns.append(step_return)
        
        asset_return = (new_price - current_price) / current_price if current_price > 0 else 0.0
        
        if self._shares_held > 0:
            action_return = asset_return
        else:
            action_return = -asset_return
            
        self._action_returns.append(action_return)
        
        self._portfolio_values.append(new_portfolio)

        # --- reward ---
        if self.reward_fn.scheme == "event_based":
            reward = 0.0
            if action == self.SELL and was_holding:
                net_sell = current_price * (1 - self.commission)
                net_buy = entry_price_before * (1 + self.commission)
                profit_pct = (net_sell - net_buy) / net_buy if net_buy > 0 else 0.0
                reward = 10.0 * profit_pct
            elif action == self.BUY and not was_holding and self._shares_held > 0:
                # Volatility over the past window_size steps as risk penalty
                start_idx = max(0, (self._current_step - 1) - self.window_size)
                end_idx = self._current_step
                past_closes = self.df["Close"].values[start_idx:end_idx]
                if len(past_closes) > 1:
                    returns_arr = np.diff(past_closes) / past_closes[:-1]
                    volatility = float(np.std(returns_arr))
                else:
                    volatility = 0.0
                reward = -0.1 * volatility
            elif action == self.HOLD:
                # if was_holding:
                #     net_sell = current_price * (1 - self.commission)
                #     net_buy = entry_price_before * (1 + self.commission)
                #     current_profit_pct = (net_sell - net_buy) / net_buy if net_buy > 0 else 0.0
                #     reward = 10.0 * 0.01 * current_profit_pct
                reward = 0.0
                # else:
                #     reward = -0.0001
            else:
                # Invalid action penalty (e.g. buying when already holding)
                reward = -0.0001
        else:
            reward = self.reward_fn.compute(
                self._returns,
                self._action_returns,
                portfolio_values=self._portfolio_values,
                done=done,
                initial_value=self.initial_balance,
            )

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, done, False, info

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _execute(self, action: int, price: float) -> None:
        if action == self.BUY and self._shares_held == 0:
            # Calculate cost per share including commission
            cost_per_share = price * (1 + self.commission)
            # Floor division ensures we only buy what we can afford (shares + commission)
            max_shares = int(self._balance // cost_per_share)
            if max_shares > 0:
                total_cost = max_shares * cost_per_share
                # Safety check: never spend more than available balance
                if total_cost > self._balance:
                    max_shares = int(self._balance / cost_per_share)
                    total_cost = max_shares * cost_per_share
                if max_shares > 0 and total_cost <= self._balance:
                    self._shares_held = max_shares
                    self._balance -= total_cost
                    self._entry_price = price
                    self._log_trade("BUY", price, max_shares)

        elif action == self.SELL and self._shares_held > 0:
            # Calculate gross revenue and deduct commission
            gross_revenue = self._shares_held * price
            commission_cost = gross_revenue * self.commission
            net_revenue = gross_revenue - commission_cost
            self._balance += net_revenue
            self._log_trade("SELL", price, self._shares_held)
            self._shares_held = 0
            self._entry_price = 0.0

        # HOLD or invalid buy/sell → no-op

    # ------------------------------------------------------------------
    # Observation building (delegated to feature builder)
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        position = 1.0 if self._shares_held > 0 else 0.0
        return self.feature_builder.build(self.df, self._current_step, position)

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of valid actions at the current step.

        [Hold, Buy, Sell]
        - Hold is always valid
        - Buy only valid when flat (no shares held)
        - Sell only valid when holding shares
        """
        return np.array(
            [
                True,  # Hold: always valid
                self._shares_held == 0,  # Buy: only when flat
                self._shares_held > 0,  # Sell: only when holding
            ],
            dtype=bool,
        )

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
            "action_mask": self.get_action_mask(),
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
