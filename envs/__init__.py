"""Register custom trading environments with Gymnasium."""

from gymnasium.envs.registration import register

register(
    id="Trading-v0",
    entry_point="envs.trading_env:TradingEnv",
)
