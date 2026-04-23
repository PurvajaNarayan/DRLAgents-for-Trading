"""Risk-aware reward shaping functions for the trading environment."""

from __future__ import annotations

from typing import ClassVar

import numpy as np


class RewardScheme:
    """
    Pluggable reward calculator.

    Schemes:
        - "simple":           raw single-step return (fraction)
        - "sharpe":           rolling Sharpe ratio (risk-adjusted)
        - "sortino":          rolling Sortino ratio (downside-risk-adjusted)
        - "action_{simple,sharpe,sortino}": same, but over action-return stream
        - "event_based":      handled inline by TradingEnv
        - "portfolio_delta":  raw dollar change per step + terminal episode PnL bonus
                              r_t = V_t - V_{t-1}
                              r_T += V_final - V_initial  (on done)
    """

    VALID_SCHEMES: ClassVar[set] = {
        "simple", "sharpe", "sortino",
        "action_simple", "action_sharpe", "action_sortino",
        "event_based",
        "portfolio_delta",
    }

    def __init__(self, scheme: str = "sharpe", lookback: int = 20) -> None:
        if scheme not in self.VALID_SCHEMES:
            raise ValueError(f"Unknown reward scheme '{scheme}'. Choose from {self.VALID_SCHEMES}")
        self.scheme = scheme
        self.lookback = lookback

    def compute(
        self,
        returns: list[float],
        action_returns: list[float] | None = None,
        *,
        portfolio_values: list[float] | None = None,
        done: bool = False,
        initial_value: float | None = None,
    ) -> float:
        """
        Compute the reward for the latest step.

        For "portfolio_delta", reads from ``portfolio_values`` (and ``initial_value``
        + ``done`` for the terminal bonus). All other schemes ignore those kwargs.
        """
        active_returns = action_returns if self.scheme.startswith("action_") else returns
        if active_returns is None:
            active_returns = returns

        if self.scheme in ["simple", "action_simple"]:
            return self._simple(active_returns)
        elif self.scheme in ["sharpe", "action_sharpe"]:
            return self._sharpe(active_returns)
        elif self.scheme in ["sortino", "action_sortino"]:
            return self._sortino(active_returns)
        elif self.scheme in ['portfolio_delta']:
            return self._portfolio_delta(portfolio_values, done, initial_value)
        else:
            raise ValueError(f"Unknown reward scheme '{self.scheme}'. Choose from {self.VALID_SCHEMES}")

    # ------------------------------------------------------------------
    # Reward implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _simple(returns: list[float]) -> float:
        """Raw single-step return."""
        return returns[-1] if returns else 0.0

    def _sharpe(self, returns: list[float]) -> float:
        """
        Rolling Sharpe ratio over the lookback window.

        Sharpe = mean(r) / std(r)
        Returns 0 if insufficient data or zero volatility.
        """
        window = self._get_window(returns)
        if len(window) < 2:
            return 0.0
        std = np.std(window)
        if std < 1e-8:
            return 0.0
        return float(np.mean(window) / std)

    def _sortino(self, returns: list[float]) -> float:
        """
        Rolling Sortino ratio over the lookback window.

        Sortino = mean(r) / downside_std(r)
        Only penalizes negative volatility, rewarding consistent upside.
        """
        window = self._get_window(returns)
        if len(window) < 2:
            return 0.0
        downside = np.array([r for r in window if r < 0])
        if len(downside) < 1:
            return float(np.mean(window)) * 10.0
        downside_std = np.std(downside)
        if downside_std < 1e-8:
            return 0.0
        return float(np.mean(window) / downside_std)

    def _portfolio_delta(
        self,
        portfolio_values: list[float] | None,
        done: bool,
        initial_value: float | None,
    ) -> float:
        """
        Raw dollar portfolio-value delta per step + terminal episode PnL bonus.

        r_t = V_curr - V_prev
        r_T += V_final - V_initial  (on terminal step)
        """
        if portfolio_values is None or len(portfolio_values) < 2:
            return 0.0

        v_curr = float(portfolio_values[-1])
        v_prev = float(portfolio_values[-2])

        reward = v_curr - v_prev

        if done and initial_value is not None:
            v_final = v_curr
            v_initial = float(initial_value)
            terminal_bonus = v_final - v_initial
            reward += terminal_bonus

        return float(reward)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_window(self, returns: list[float]) -> np.ndarray:
        """Slice the most recent `lookback` returns."""
        return np.array(returns[-self.lookback :])
