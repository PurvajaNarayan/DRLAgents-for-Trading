"""Smoke tests for the TradingEnv and RewardScheme."""

import numpy as np
import pandas as pd
import pytest

from envs.rewards import RewardScheme
from envs.trading import TradingEnv

# ----- fixtures -----


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Generate 100 rows of synthetic OHLCV data."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 0.1,
            "High": close + abs(np.random.randn(n) * 0.3),
            "Low": close - abs(np.random.randn(n) * 0.3),
            "Close": close,
            "Volume": np.random.randint(1000, 10000, size=n),
        }
    )


@pytest.fixture
def env(sample_df) -> TradingEnv:
    return TradingEnv(df=sample_df, window_size=10, initial_balance=10_000.0)


# ----- TradingEnv tests -----


class TestTradingEnv:
    def test_reset_returns_correct_shape(self, env):
        obs, _info = env.reset(seed=0)
        expected_dim = env.window_size * 5 + 1
        assert obs.shape == (expected_dim,)
        assert obs.dtype == np.float32

    def test_reset_initializes_balance(self, env):
        _, info = env.reset()
        assert info["balance"] == 10_000.0
        assert info["shares_held"] == 0

    def test_step_hold_does_not_change_shares(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(TradingEnv.HOLD)
        assert info["shares_held"] == 0

    def test_buy_then_sell_round_trip(self, env):
        env.reset(seed=0)
        env.step(TradingEnv.BUY)
        _, _, _, _, info_after_sell = env.step(TradingEnv.SELL)
        assert info_after_sell["shares_held"] == 0
        # balance changes due to price movement + commission
        assert info_after_sell["balance"] != 10_000.0

    def test_episode_terminates(self, env):
        env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            _, _, done, _, _ = env.step(env.action_space.sample())
            steps += 1
        assert steps > 0

    def test_invalid_df_raises(self):
        bad_df = pd.DataFrame({"Close": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing required columns"):
            TradingEnv(df=bad_df)

    def test_observation_normalized_0_to_1(self, env):
        obs, _ = env.reset(seed=0)
        ohlcv_part = obs[:-1]  # exclude position flag
        assert ohlcv_part.min() >= -0.01  # small float tolerance
        assert ohlcv_part.max() <= 1.01


# ----- RewardScheme tests -----


class TestRewardScheme:
    def test_simple_returns_last(self):
        rs = RewardScheme(scheme="simple")
        assert rs.compute([0.01, 0.02, -0.005]) == pytest.approx(-0.005)

    def test_sharpe_zero_on_empty(self):
        rs = RewardScheme(scheme="sharpe")
        assert rs.compute([]) == 0.0

    def test_sharpe_zero_on_constant_returns(self):
        rs = RewardScheme(scheme="sharpe", lookback=5)
        assert rs.compute([0.01] * 10) == 0.0  # std=0 → 0

    def test_sortino_rewards_pure_upside(self):
        rs = RewardScheme(scheme="sortino", lookback=10)
        reward = rs.compute([0.01] * 10)
        assert reward > 0  # no downside → positive reward

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="Unknown reward scheme"):
            RewardScheme(scheme="invalid")
