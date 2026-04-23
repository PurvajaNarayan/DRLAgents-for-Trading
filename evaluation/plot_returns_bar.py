"""
Bar chart of cumulative returns for DQN, PPO, LSTM-PPO vs Buy & Hold.
Matches the style: blue shaded bars, red dashed B&H line, % labels.

Usage:
    python -m evaluation.plot_returns_bar --ticker CHWY --split test
    python -m evaluation.plot_returns_bar --ticker CHWY --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from envs.trading import TradingEnv
from features import OHLCVWithIndicators, RawOHLCV


# ── agent loader ────────────────────────────────────────────────────────────

def load_agent(agent_type: str, obs_dim: int, act_dim: int, ckpt_path: Path):
    if agent_type == "ppo":
        from agents.policy_gradient.ppo import PPOAgent
        cfg = {"gamma": 0.99, "lr": 3e-4, "hidden": 128,
               "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64}
        a = PPOAgent(obs_dim, act_dim, cfg); a.load(ckpt_path); return a

    if agent_type == "dqn":
        from agents.value_based.dqn import DQNAgent
        cfg = {"gamma": 0.99, "lr": 5e-4, "hidden": (128, 64, 32),
               "double_dqn": True, "epsilon_start": 0.0, "epsilon_min": 0.0}
        a = DQNAgent(obs_dim, act_dim, cfg); a.load(ckpt_path); return a

    if agent_type == "pomdp":
        from agents.partial_mdp.agent import PMDPAgent
        cfg = {"gamma": 0.99, "lr": 3e-4, "lr_lstm": 1e-3, "hidden": 128,
               "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64}
        a = PMDPAgent(obs_dim, act_dim, cfg); a.load(ckpt_path); return a

    raise ValueError(f"Unknown agent: {agent_type}")


# ── evaluation helpers ───────────────────────────────────────────────────────

def cumulative_return(env: TradingEnv, agent, agent_type: str) -> float:
    """Return (final_value - initial_value) / initial_value."""
    obs, info = env.reset()
    done = False
    initial = info["portfolio_value"]

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
        obs, _, done, _, info = env.step(action)

    final = info["portfolio_value"]
    return (final - initial) / initial


def buy_and_hold_return(env: TradingEnv) -> float:
    obs, info = env.reset()
    obs, _, done, _, info = env.step(TradingEnv.BUY)
    initial = info["portfolio_value"]
    while not done:
        obs, _, done, _, info = env.step(TradingEnv.HOLD)
    return (info["portfolio_value"] - initial) / initial


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",      default="CHWY")
    parser.add_argument("--reward",      default="event_based")
    parser.add_argument("--features",    default="indicators")
    parser.add_argument("--split",       choices=["val", "test"], default="test")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--save_path",   default=None)
    args = parser.parse_args()

    eval_df = pd.read_csv(
        f"data/processed/{args.ticker}_{args.split}.csv",
        index_col=0, parse_dates=True,
    )

    # Auto-discover available checkpoints — no hardcoding
    REGISTRY = [
        ("DQN",      "dqn",   "double_dqn", "dqn.pt"),
        ("PPO",      "ppo",   "ppo",         "ppo.pt"),
        ("LSTM-PPO", "pomdp", "pmdp",        "pmdp.pt"),
    ]

    AGENTS = []
    for label, atype, prefix, mfile in REGISTRY:
        folder = f"{prefix}_{args.ticker}_{args.features}_{args.reward}"
        ckpt = Path("runs") / folder
        if (ckpt / mfile).exists():
            # infer obs_dim from checkpoint to get correct window size
            import torch
            try:
                if atype == "dqn":
                    sd = torch.load(ckpt / mfile, weights_only=True)
                    obs_dim = sd["q_net"][next(iter(sd["q_net"]))].shape[1]
                elif atype == "ppo":
                    sd = torch.load(ckpt / mfile, weights_only=True)
                    obs_dim = sd["policy"][next(iter(sd["policy"]))].shape[1]
                elif atype == "pomdp":
                    sd = torch.load(ckpt / mfile, weights_only=True)
                    # actor first layer: input = obs_dim + hidden_dim (128)
                    in_dim = sd["actor_critic"][next(iter(sd["actor_critic"]))].shape[1]
                    obs_dim = in_dim - 128
                win = (obs_dim - 7) // 5  # reverse: obs_dim = W*5 + 6 + 1
            except Exception:
                win = args.window_size
            AGENTS.append((label, atype, folder, mfile, win))
            print(f"  Found [{label}] — {folder}  (inferred window={win})")
        else:
            print(f"  Skip  [{label}] — {folder}/{mfile} not found")

    labels, returns = [], []

    for label, atype, folder, mfile, win in AGENTS:
        ckpt = Path("runs") / folder
        if not (ckpt / mfile).exists():
            print(f"[{label}] checkpoint not found — skipping"); continue

        fb = OHLCVWithIndicators(window_size=win) if args.features == "indicators" else RawOHLCV(window_size=win)
        env = TradingEnv(df=eval_df, feature_builder=fb, window_size=win,
                         reward_scheme=args.reward, max_episode_steps=None)

        try:
            agent = load_agent(atype, env.observation_space.shape[0], env.action_space.n, ckpt)
            ret = cumulative_return(env, agent, atype)
            labels.append(label)
            returns.append(ret * 100)
            print(f"  [{label:>8s}]  {ret:+.2%}")
        except Exception as e:
            print(f"  [{label}] error: {e} — skipping")

    # buy & hold (use standard window)
    fb_bh = OHLCVWithIndicators(window_size=args.window_size)
    env_bh = TradingEnv(df=eval_df, feature_builder=fb_bh, window_size=args.window_size,
                        reward_scheme=args.reward, max_episode_steps=None)
    bh = buy_and_hold_return(env_bh) * 100
    print(f"  [Buy&Hold]  {bh:+.2f}%")

    # ── plot ──────────────────────────────────────────────────────────────────
    bar_colors = ["#1a3a5c", "#2e6b9e", "#5ba3d9"][:len(labels)]

    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.arange(len(labels))
    bars = ax.bar(x, returns, width=0.45,
                  color=bar_colors, edgecolor="white", linewidth=0.5)

    # buy & hold dashed line
    ax.axhline(y=bh, color="#e74c3c", lw=2, ls="--", zorder=4,
               label=f"Buy & Hold ({bh:+.1f}%)")

    # value labels on bars
    for bar, val in zip(bars, returns):
        offset = 4 if val >= 0 else -14
        ax.annotate(
            f"{val:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center", fontsize=12, fontweight="bold",
            color="#1a3a5c",
        )

    ax.set_title(
        f"{args.ticker} {args.split.title()} Set — Cumulative Return",
        fontsize=13, fontweight="bold", color="#1a3a5c",
    )
    ax.set_ylabel("Cumulative Return (%)", fontsize=11, color="#1a3a5c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(colors="#1a3a5c")
    ax.axhline(y=0, color="#1a3a5c", lw=0.5)
    ax.grid(alpha=0.15, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.0f%%"))

    plt.tight_layout()

    save = args.save_path or f"images/returns_bar_{args.ticker}_{args.split}.png"
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save}")
    plt.show()
