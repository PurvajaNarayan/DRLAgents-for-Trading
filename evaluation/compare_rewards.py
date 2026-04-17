"""Compare cumulative rewards across different reward functions on the test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents.policy_gradient.ppo import PPOAgent
from envs.trading import TradingEnv
from evaluation.plots import plot_reward_comparison
from features import OHLCVWithIndicators, RawOHLCV


def collect_rewards(env: TradingEnv, agent: PPOAgent) -> list[float]:
    """Run one episode with greedy actions and return per-step rewards."""
    obs, info = env.reset()
    done = False
    rewards = []

    while not done:
        mask = info.get("action_mask")
        action, _, _ = agent.select_action(obs, explore=False, action_mask=mask)
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)

    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare reward functions")
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--features", choices=["raw", "indicators"], default="indicators")
    args = parser.parse_args()

    # --- load test data ---
    test_df = pd.read_csv(f"data/processed/{args.ticker}_test.csv", index_col=0, parse_dates=True)

    fb = RawOHLCV(window_size=20) if args.features == "raw" else OHLCVWithIndicators(window_size=20)

    # --- run each reward function ---
    reward_schemes = ["simple", "sharpe", "sortino"]
    checkpoint_dir = f"runs/ppo_{args.ticker}_{args.features}"

    all_rewards = {}

    for scheme in reward_schemes:
        ckpt_path = Path(f"runs/ppo_{args.ticker}_{args.features}_{scheme}")

        if not (ckpt_path / "ppo.pt").exists():
            print(f"Checkpoint not found: {ckpt_path}/ppo.pt — skipping {scheme}")
            continue

        env = TradingEnv(df=test_df, feature_builder=fb, reward_scheme=scheme)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        config = {
            "gamma": 0.99,
            "lr": 3e-4,
            "hidden": 128,
            "clip_eps": 0.2,
            "n_epochs": 10,
            "batch_size": 64,
            "entropy_coef": 0.05,
        }
        agent = PPOAgent(obs_dim, act_dim, config)
        agent.load(ckpt_path)

        rewards = collect_rewards(env, agent)
        all_rewards[scheme.capitalize()] = rewards
        print(
            f"{scheme.capitalize():>8s} | Steps: {len(rewards)} | Total reward: {sum(rewards):.4f}"
        )

    # --- plot ---
    if all_rewards:
        save_dir = f"runs/ppo_{args.ticker}_{args.features}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        plot_reward_comparison(
            results=all_rewards,
            title=f"Cumulative Reward — {args.ticker} (Test Set)",
            save_path=f"{save_dir}/reward_comparison.png",
        )
    else:
        print("No checkpoints found. Train with each reward function first.")
        print("Example:")
        print(
            f"  python -m agents.policy_gradient.train --ticker {args.ticker} --features {args.features} --reward simple --episodes 150"
        )
        print(
            f"  python -m agents.policy_gradient.train --ticker {args.ticker} --features {args.features} --reward sharpe --episodes 150"
        )
        print(
            f"  python -m agents.policy_gradient.train --ticker {args.ticker} --features {args.features} --reward sortino --episodes 150"
        )
