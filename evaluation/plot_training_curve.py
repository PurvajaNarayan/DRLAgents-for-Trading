"""
Plot episode reward vs episode number for trained agents.
Re-runs training with history logging and saves the curve.

Usage:
    python -m evaluation.plot_training_curve --agent dqn --ticker CHWY
    python -m evaluation.plot_training_curve --agent ppo --ticker CHWY
    python -m evaluation.plot_training_curve --agent all --ticker CHWY
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def smooth(values: list[float], window: int = 20) -> np.ndarray:
    """Rolling mean smoothing."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def train_and_collect_dqn(ticker: str, reward: str, features: str, episodes: int, window_size: int):
    """Train DQN and return per-episode reward history."""
    from agents.value_based.dqn import DQNAgent
    from agents.value_based.train import train_dqn
    from envs.trading import TradingEnv
    from features import OHLCVWithIndicators, RawOHLCV

    train_df = pd.read_csv(
        f"data/processed/{ticker}_train.csv", index_col=0, parse_dates=True
    )
    fb = (
        RawOHLCV(window_size=window_size)
        if features == "raw"
        else OHLCVWithIndicators(window_size=window_size)
    )
    env = TradingEnv(
        df=train_df,
        feature_builder=fb,
        window_size=window_size,
        reward_scheme=reward,
        max_episode_steps=252,
    )
    config = {
        "gamma": 0.99,
        "lr": 5e-4,
        "batch_size": 64,
        "buffer_size": 50_000,
        "target_update_freq": 500,
        "train_start": 500,
        "epsilon_start": 0.3,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "double_dqn": True,
        "hidden": (128, 64, 32),
    }
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, config)
    history = train_dqn(env, agent, n_episodes=episodes, log_interval=episodes + 1, plot_every=episodes + 1)
    return [h["ep_reward"] for h in history]


def train_and_collect_ppo(ticker: str, reward: str, features: str, episodes: int, window_size: int):
    """Train PPO and return per-episode reward history."""
    from agents.policy_gradient.ppo import PPOAgent
    from agents.policy_gradient.train import train_ppo
    from envs.trading import TradingEnv
    from features import OHLCVWithIndicators, RawOHLCV

    train_df = pd.read_csv(
        f"data/processed/{ticker}_train.csv", index_col=0, parse_dates=True
    )
    fb = (
        RawOHLCV(window_size=window_size)
        if features == "raw"
        else OHLCVWithIndicators(window_size=window_size)
    )
    env = TradingEnv(
        df=train_df,
        feature_builder=fb,
        window_size=window_size,
        reward_scheme=reward,
        max_episode_steps=252,
    )
    config = {
        "gamma": 0.99,
        "lr": 3e-4,
        "hidden": 128,
        "clip_eps": 0.2,
        "n_epochs": 10,
        "batch_size": 64,
    }
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, config)
    history = train_ppo(env, agent, n_episodes=episodes, rollout_steps=64, log_interval=episodes + 1, plot_every=episodes + 1)
    return [h["ep_reward"] for h in history]


def train_and_collect_pomdp(ticker: str, reward: str, features: str, episodes: int, window_size: int):
    """Train POMDP (LSTM-PPO) and return per-episode reward history."""
    from agents.partial_mdp.agent import PMDPAgent
    from agents.partial_mdp.train import pretrain_lstm, train_pmdp_ppo
    from envs.trading import TradingEnv
    from features import OHLCVWithIndicators, RawOHLCV

    train_df = pd.read_csv(
        f"data/processed/{ticker}_train.csv", index_col=0, parse_dates=True
    )
    fb = (
        RawOHLCV(window_size=window_size)
        if features == "raw"
        else OHLCVWithIndicators(window_size=window_size)
    )
    env = TradingEnv(
        df=train_df,
        feature_builder=fb,
        window_size=window_size,
        reward_scheme=reward,
        max_episode_steps=252,
    )
    config = {
        "gamma": 0.99,
        "lr": 3e-4,
        "lr_lstm": 1e-3,
        "hidden": 128,
        "clip_eps": 0.2,
        "n_epochs": 10,
        "batch_size": 64,
    }
    agent = PMDPAgent(env.observation_space.shape[0], env.action_space.n, config)
    pretrain_lstm(env, agent, n_episodes=50)
    history = train_pmdp_ppo(env, agent, n_episodes=episodes, rollout_steps=64, plot_every=episodes + 1)
    return [h["ep_reward"] for h in history]


def plot_curves(
    curves: dict[str, list[float]],
    ticker: str,
    smooth_window: int = 20,
    save_path: str | None = None,
) -> None:
    """Plot one smoothed reward curve per agent on a single axes."""
    colors = {
        "Double DQN": "#e74c3c",
        "PPO":        "#3498db",
        "LSTM-PPO":   "#2ecc71",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, rewards in curves.items():
        episodes = np.arange(1, len(rewards) + 1)
        raw = np.array(rewards, dtype=float)
        smoothed = smooth(rewards, window=smooth_window)

        color = colors.get(name, "#999999")
        ax.plot(episodes, raw, color=color, alpha=0.15, lw=1)
        ax.plot(episodes, smoothed, color=color, lw=2.2, label=f"{name} (smoothed)")

    ax.axhline(y=0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(
        f"Episode Reward vs Episodes — {ticker}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Episode Reward", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot episode reward vs episode for DQN / PPO / LSTM-PPO")
    parser.add_argument("--agent", choices=["dqn", "ppo", "pomdp", "all"], default="all")
    parser.add_argument("--ticker", default="CHWY")
    parser.add_argument("--reward", default="event_based")
    parser.add_argument("--features", choices=["raw", "indicators"], default="indicators")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--smooth", type=int, default=20, help="Rolling window for smoothing")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save figure (e.g. images/reward_curve.png)")
    args = parser.parse_args()

    curves: dict[str, list[float]] = {}

    if args.agent in ("dqn", "all"):
        print("Training Double DQN...")
        curves["Double DQN"] = train_and_collect_dqn(
            args.ticker, args.reward, args.features, args.episodes, args.window_size
        )

    if args.agent in ("ppo", "all"):
        print("Training PPO...")
        curves["PPO"] = train_and_collect_ppo(
            args.ticker, args.reward, args.features, args.episodes, args.window_size
        )

    if args.agent in ("pomdp", "all"):
        print("Training LSTM-PPO...")
        curves["LSTM-PPO"] = train_and_collect_pomdp(
            args.ticker, args.reward, args.features, args.episodes, args.window_size
        )

    save = args.save_path or f"images/reward_curve_{args.ticker}_{args.agent}.png"
    plot_curves(curves, ticker=args.ticker, smooth_window=args.smooth, save_path=save)
