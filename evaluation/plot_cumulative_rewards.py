"""Plot cumulative rewards for DQN, PPO, and LSTM-PPO on a single graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.trading import TradingEnv
from features import OHLCVWithIndicators, RawOHLCV


def collect_rewards(env: TradingEnv, agent, agent_type: str) -> list[float]:
    """Run one greedy episode and return per-step rewards."""
    obs, info = env.reset()
    done = False
    rewards = []

    if agent_type == "pomdp":
        agent.reset_hidden_state()

    while not done:
        mask = info.get("action_mask")

        if agent_type == "ppo":
            action, _, _ = agent.select_action(obs, explore=False, action_mask=mask)
        elif agent_type == "pomdp":
            action, _, _ = agent.select_action(obs, action_mask=mask)
            agent.update_hidden_state(obs)
        else:
            action = agent.select_action(obs, explore=False, action_mask=mask)

        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)

    return rewards


def load_agent(agent_type: str, obs_dim: int, act_dim: int, ckpt_path: Path):
    if agent_type == "ppo":
        from agents.policy_gradient.ppo import PPOAgent
        config = {"gamma": 0.99, "lr": 3e-4, "hidden": 128,
                  "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64}
        agent = PPOAgent(obs_dim, act_dim, config)
        agent.load(ckpt_path)
        return agent

    elif agent_type == "dqn":
        from agents.value_based.dqn import DQNAgent
        config = {"gamma": 0.99, "lr": 5e-4, "hidden": (128, 64, 32),
                  "double_dqn": True, "epsilon_start": 0.0, "epsilon_min": 0.0}
        agent = DQNAgent(obs_dim, act_dim, config)
        agent.load(ckpt_path)
        return agent

    elif agent_type == "pomdp":
        from agents.partial_mdp.agent import PMDPAgent
        config = {"gamma": 0.99, "lr": 3e-4, "lr_lstm": 1e-3, "hidden": 128,
                  "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64}
        agent = PMDPAgent(obs_dim, act_dim, config)
        agent.load(ckpt_path)
        return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   default="CHWY")
    parser.add_argument("--reward",   default="event_based")
    parser.add_argument("--features", default="indicators")
    parser.add_argument("--split",    choices=["val", "test"], default="test")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--dqn_window",  type=int, default=30,
                        help="Window size used when DQN was trained (may differ)")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    AGENTS = [
        ("Double DQN", "dqn",   f"double_dqn_{args.ticker}_{args.features}_{args.reward}", "dqn.pt",  args.dqn_window, "#e74c3c"),
        ("PPO",        "ppo",   f"ppo_{args.ticker}_{args.features}_{args.reward}",         "ppo.pt",  args.window_size, "#3498db"),
        ("LSTM-PPO",   "pomdp", f"pmdp_{args.ticker}_{args.features}_{args.reward}",        "pmdp.pt", args.window_size, "#2ecc71"),
    ]

    eval_df = pd.read_csv(
        f"data/processed/{args.ticker}_{args.split}.csv",
        index_col=0, parse_dates=True
    )
    print(f"{args.split.upper()} set: {len(eval_df)} days of {args.ticker}\n")

    fig, ax = plt.subplots(figsize=(11, 5))

    for label, agent_type, folder, model_file, win, color in AGENTS:
        ckpt_path = Path("runs") / folder
        if not (ckpt_path / model_file).exists():
            print(f"  [{label}] checkpoint not found — skipping")
            continue

        fb = (RawOHLCV(window_size=win)
              if args.features == "raw"
              else OHLCVWithIndicators(window_size=win))

        env = TradingEnv(
            df=eval_df,
            feature_builder=fb,
            window_size=win,
            reward_scheme=args.reward,
            max_episode_steps=None,
        )
        obs_dim = int(env.observation_space.shape[0])
        act_dim  = int(env.action_space.n)

        try:
            agent = load_agent(agent_type, obs_dim, act_dim, ckpt_path)
        except Exception as e:
            print(f"  [{label}] load error: {e} — skipping")
            continue

        rewards   = collect_rewards(env, agent, agent_type)
        cum       = np.cumsum(rewards)
        days      = np.arange(1, len(cum) + 1)
        total     = cum[-1] if len(cum) else 0.0

        ax.plot(days, cum, color=color, lw=2.2,
                label=f"{label}  (total={total:+.3f})")
        print(f"  [{label:>10s}] steps={len(rewards):3d} | cumulative reward={total:+.4f}")

    ax.axhline(y=0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_title(
        f"Cumulative Reward — DQN vs PPO vs LSTM-PPO\n"
        f"{args.ticker} · {args.split.title()} Set · {args.reward.replace('_',' ').title()} Reward",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Trading Day", fontsize=11)
    ax.set_ylabel("Cumulative Reward", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    save = args.save_path or f"images/cumulative_reward_{args.ticker}_{args.split}.png"
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save}")
    plt.show()
