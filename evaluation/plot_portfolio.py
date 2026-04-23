"""
Plot portfolio value trajectory on the test set for all agents side by side
against the buy-and-hold baseline.

This is the single most persuasive plot in a trading RL paper because it shows
not just the final return but the *path* the agent took to get there.

Usage:
    python plot_portfolio_trajectory.py --ticker AMZN --split test
    python plot_portfolio_trajectory.py --ticker CHWY --split test

Assumes checkpoints live at:
    runs/double_dqn_{TICKER}_indicators_event_based/
    runs/ppo_{TICKER}_indicators_event_based/
    runs/pmdp_{TICKER}_indicators_event_based/
    runs/rainbow_{TICKER}_indicators_event_based/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.partial_mdp.agent import PMDPAgent
from agents.policy_gradient.ppo import PPOAgent
from agents.value_based.dqn import DQNAgent
from agents.value_based.rainbow import RainbowAgent
from envs.trading import TradingEnv
from features import OHLCVWithIndicators


# ---------------------------------------------------------------------------
# Agent rollout (captures per-step portfolio value)
# ---------------------------------------------------------------------------


def rollout_and_track(env: TradingEnv, agent, agent_type: str) -> tuple[list[float], list[dict]]:
    """
    Run the agent greedily through one full episode and return the per-step
    portfolio value trajectory plus the trade log for overlay markers.
    """
    obs, info = env.reset()
    if agent_type == "pmdp":
        agent.reset_hidden_state()

    done = False
    while not done:
        mask = info.get("action_mask")
        if agent_type == "ppo":
            action, _, _ = agent.select_action(obs, explore=False, action_mask=mask)
        elif agent_type == "pmdp":
            action, _, _ = agent.select_action(obs, explore=False, action_mask=mask)
        elif agent_type == "rainbow":
            action = agent.select_action(obs, explore=False, action_mask=mask)
        else:  # dqn
            action = agent.select_action(obs, explore=False, action_mask=mask)

        next_obs, _, done, _, info = env.step(action)
        if agent_type == "pmdp":
            agent.update_hidden_state(obs)
        obs = next_obs

    return list(env._portfolio_values), list(info["trade_log"])


def buy_and_hold_trajectory(env: TradingEnv) -> list[float]:
    """Replay buy-and-hold and capture its portfolio value trajectory."""
    _obs, info = env.reset()
    _obs, _, _, _, info = env.step(TradingEnv.BUY)
    done = False
    while not done:
        _obs, _, done, _, info = env.step(TradingEnv.HOLD)
    return list(env._portfolio_values)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_trajectories(
    trajectories: dict[str, list[float]],
    baseline: list[float],
    initial_balance: float,
    ticker: str,
    split: str,
    save_path: str,
) -> None:
    """
    Plot cumulative return (%) over trading days, one line per agent plus baseline.
    We use % return (not $) so different tickers are comparable.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))

    colors = {
        "Double DQN": "#c0392b",   # red
        "Rainbow":    "#9b59b6",   # purple
        "PPO":        "#2980b9",   # blue
        "LSTM-PPO":   "#27ae60",   # green
    }

    # agent lines
    for name, traj in trajectories.items():
        traj_arr = np.asarray(traj)
        pct = (traj_arr / initial_balance - 1.0) * 100.0
        ax.plot(pct, lw=2.0, label=name, color=colors.get(name, "gray"))

    # baseline
    bh_arr = np.asarray(baseline)
    bh_pct = (bh_arr / initial_balance - 1.0) * 100.0
    ax.plot(
        bh_pct, lw=2.0, ls="--", color="#34495e", label="Buy & Hold",
    )

    # cosmetics
    ax.axhline(y=0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("Trading Day", fontsize=11)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax.set_title(
        f"{ticker} — Portfolio Value Over Time ({split.title()} Set)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10, framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AMZN")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--save_dir", default="runs/analysis_plots")
    # Rainbow params
    parser.add_argument("--n_atoms", type=int, default=51)
    parser.add_argument("--v_min", type=float, default=-10.0)
    parser.add_argument("--v_max", type=float, default=10.0)
    parser.add_argument("--hidden_rainbow", type=int, default=64)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    df = pd.read_csv(
        f"data/processed/{args.ticker}_{args.split}.csv",
        index_col=0, parse_dates=True,
    )
    fb = OHLCVWithIndicators(window_size=args.window_size)

    def fresh_env() -> TradingEnv:
        return TradingEnv(
            df=df, feature_builder=fb,
            window_size=args.window_size,
            reward_scheme="event_based",
            max_episode_steps=None,
        )

    env = fresh_env()
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    # --- helper: infer obs_dim and window from checkpoint ---
    import torch

    def infer_window(ckpt_path: Path, agent_type: str) -> int:
        """Read first-layer weight shape to back out the window size used at training."""
        try:
            if agent_type == "dqn":
                sd = torch.load(ckpt_path / "dqn.pt", weights_only=True)
                in_dim = sd["q_net"][next(iter(sd["q_net"]))].shape[1]
            elif agent_type == "rainbow":
                sd = torch.load(ckpt_path / "rainbow.pt", weights_only=True)
                in_dim = sd["network"][next(iter(sd["network"]))].shape[1]
            elif agent_type == "ppo":
                sd = torch.load(ckpt_path / "ppo.pt", weights_only=True)
                in_dim = sd["policy"][next(iter(sd["policy"]))].shape[1]
            elif agent_type == "pmdp":
                sd = torch.load(ckpt_path / "pmdp.pt", weights_only=True)
                in_dim = sd["actor_critic"][next(iter(sd["actor_critic"]))].shape[1] - 128
            else:
                return args.window_size
            return (in_dim - 7) // 5   # reverse: obs_dim = W*5 + 6 + 1
        except Exception:
            return args.window_size

    def fresh_env_for(win: int) -> TradingEnv:
        return TradingEnv(
            df=df,
            feature_builder=OHLCVWithIndicators(window_size=win),
            window_size=win,
            reward_scheme="event_based",
            max_episode_steps=None,
        )

    # --- load agents ---
    trajectories: dict[str, list[float]] = {}

    # Double DQN
    dqn_ckpt = Path(f"runs/double_dqn_{args.ticker}_indicators_event_based")
    try:
        win = infer_window(dqn_ckpt, "dqn")
        env_dqn = fresh_env_for(win)
        dqn_obs_dim = int(env_dqn.observation_space.shape[0])
        dqn_cfg = {"hidden": (128, 64, 32), "double_dqn": True,
                   "epsilon_start": 0.0, "epsilon_min": 0.0}
        dqn = DQNAgent(dqn_obs_dim, act_dim, dqn_cfg)
        dqn.load(dqn_ckpt)
        traj, _ = rollout_and_track(env_dqn, dqn, "dqn")
        trajectories["Double DQN"] = traj
        print(f"Double DQN  (win={win}): final = ${traj[-1]:,.2f}")
    except Exception as e:
        print(f"Skipping Double DQN: {e}")

    # Rainbow
    rb_ckpt = Path(f"runs/rainbow_{args.ticker}_indicators_event_based")
    try:
        win = infer_window(rb_ckpt, "rainbow")
        env_rb = fresh_env_for(win)
        rb_obs_dim = int(env_rb.observation_space.shape[0])
        sd = torch.load(rb_ckpt / "rainbow.pt", weights_only=True)
        rb_hidden = sd["network"][next(iter(sd["network"]))].shape[0]
        rb_cfg = {
            "n_atoms": args.n_atoms, "v_min": args.v_min, "v_max": args.v_max,
            "hidden": rb_hidden,
        }
        rainbow = RainbowAgent(rb_obs_dim, act_dim, rb_cfg)
        rainbow.load(rb_ckpt)
        traj, _ = rollout_and_track(env_rb, rainbow, "rainbow")
        trajectories["Rainbow"] = traj
        print(f"Rainbow     (win={win}): final = ${traj[-1]:,.2f}")
    except Exception as e:
        print(f"Skipping Rainbow: {e}")

    # PPO
    ppo_ckpt = Path(f"runs/ppo_{args.ticker}_indicators_event_based")
    try:
        win = infer_window(ppo_ckpt, "ppo")
        env_ppo = fresh_env_for(win)
        ppo_obs_dim = int(env_ppo.observation_space.shape[0])
        ppo_cfg = {"gamma": 0.99, "lr": 3e-4, "hidden": 128, "clip_eps": 0.2,
                   "n_epochs": 10, "batch_size": 64}
        ppo = PPOAgent(ppo_obs_dim, act_dim, ppo_cfg)
        ppo.load(ppo_ckpt)
        traj, _ = rollout_and_track(env_ppo, ppo, "ppo")
        trajectories["PPO"] = traj
        print(f"PPO         (win={win}): final = ${traj[-1]:,.2f}")
    except Exception as e:
        print(f"Skipping PPO: {e}")

    # LSTM-PPO (PMDP)
    pmdp_ckpt = Path(f"runs/pmdp_{args.ticker}_indicators_event_based")
    try:
        win = infer_window(pmdp_ckpt, "pmdp")
        env_pmdp = fresh_env_for(win)
        pmdp_obs_dim = int(env_pmdp.observation_space.shape[0])
        pmdp_cfg = {"gamma": 0.99, "lr": 3e-4, "lr_lstm": 1e-3, "hidden": 128,
                    "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64}
        pmdp = PMDPAgent(pmdp_obs_dim, act_dim, pmdp_cfg)
        pmdp.load(pmdp_ckpt)
        pmdp.freeze_predictor()
        traj, _ = rollout_and_track(env_pmdp, pmdp, "pmdp")
        trajectories["LSTM-PPO"] = traj
        print(f"LSTM-PPO    (win={win}): final = ${traj[-1]:,.2f}")
    except Exception as e:
        print(f"Skipping LSTM-PPO: {e}")

    # Buy & Hold baseline
    env = fresh_env_for(args.window_size)
    bh = buy_and_hold_trajectory(env)
    print(f"Buy & Hold: final = ${bh[-1]:,.2f}")

    # --- plot ---
    plot_trajectories(
        trajectories, bh,
        initial_balance=env.initial_balance,
        ticker=args.ticker, split=args.split,
        save_path=str(save_dir / f"trajectory_{args.ticker}_{args.split}.png"),
    )