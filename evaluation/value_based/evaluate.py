"""Evaluate a trained value-based (DQN / Double DQN / Rainbow) agent on val or test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agents.base import BaseAgent
from agents.value_based.dqn import DQNAgent
from agents.value_based.rainbow import RainbowAgent
from envs.trading import TradingEnv
from features import OHLCVWithIndicators, RawOHLCV


def evaluate(env: TradingEnv, agent: BaseAgent, n_episodes: int = 1):
    """
    Run a value-based agent on a data split with NO exploration.

    For DQN, this means `agent.epsilon = 0.0` (pure argmax over Q).
    For Rainbow, `agent.select_action(..., explore=False)` internally
    switches the NoisyLinear layers to deterministic (mean-only) mode.
    """
    saved_epsilon = getattr(agent, "epsilon", 0.0)
    agent.epsilon = 0.0

    results = []

    try:
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            daily_returns: list[float] = []

            initial_value = info["portfolio_value"]

            while not done:
                mask = info.get("action_mask")
                action = agent.select_action(obs, explore=False, action_mask=mask)

                prev_pv = info["portfolio_value"]
                obs, reward, done, _, info = env.step(action)
                ep_reward += reward

                curr_pv = info["portfolio_value"]
                daily_ret = (curr_pv - prev_pv) / prev_pv if prev_pv > 0 else 0.0
                daily_returns.append(daily_ret)

            final_value = info["portfolio_value"]
            returns_arr = np.array(daily_returns)

            cumulative_return = (final_value - initial_value) / initial_value
            sharpe = (
                np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)
                if np.std(returns_arr) > 1e-8
                else 0.0
            )

            downside = returns_arr[returns_arr < 0]
            sortino = (
                np.mean(returns_arr) / np.std(downside) * np.sqrt(252)
                if len(downside) > 0 and np.std(downside) > 1e-8
                else 0.0
            )

            portfolio_values = np.array(env._portfolio_values)
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = float(np.max(drawdown))

            trades = info["trade_log"]
            n_trades = len([t for t in trades if t["side"] == "SELL"])

            results.append(
                {
                    "episode": ep,
                    "initial_value": initial_value,
                    "final_value": final_value,
                    "cumulative_return": cumulative_return,
                    "sharpe_ratio": sharpe,
                    "sortino_ratio": sortino,
                    "max_drawdown": max_drawdown,
                    "total_trades": n_trades,
                    "ep_reward": ep_reward,
                }
            )
    finally:
        agent.epsilon = saved_epsilon

    return results


def buy_and_hold_baseline(env: TradingEnv) -> dict:
    """Buy on day 1, hold until the end. Simplest benchmark."""
    _obs, info = env.reset()
    _obs, _reward, done, _, info = env.step(TradingEnv.BUY)
    initial_value = info["portfolio_value"]

    while not done:
        _obs, _reward, done, _, info = env.step(TradingEnv.HOLD)

    final_value = info["portfolio_value"]
    return {
        "strategy": "Buy & Hold",
        "initial_value": initial_value,
        "final_value": final_value,
        "cumulative_return": (final_value - initial_value) / initial_value,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN / Double DQN / Rainbow agent on val or test split"
    )
    parser.add_argument(
        "--agent",
        choices=["dqn", "rainbow"],
        default="dqn",
        help="Which value-based agent class to instantiate before loading the checkpoint.",
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--features", choices=["raw", "indicators"], default="indicators")
    parser.add_argument(
        "--reward",
        choices=["simple", "sharpe", "sortino", "event_based", "portfolio_delta"],
        default="portfolio_delta",
    )
    parser.add_argument(
        "--split", choices=["val", "test"], default="test",
        help="Which data split to evaluate on.",
    )
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Folder containing dqn.pt or rainbow.pt "
             "(e.g. runs/rainbow_AAPL_indicators_portfolio_delta).",
    )
    parser.add_argument(
        "--double_dqn",
        action="store_true",
        help="(dqn only) Set this to match the variant used at training time.",
    )
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting trading behavior.")

    # Rainbow reconstruction params: must match training time (shape is tied to n_atoms/hidden).
    parser.add_argument("--n_atoms", type=int, default=51, help="(rainbow) atoms in C51.")
    parser.add_argument("--v_min", type=float, default=-10.0, help="(rainbow) support min.")
    parser.add_argument("--v_max", type=float, default=10.0, help="(rainbow) support max.")
    parser.add_argument("--hidden_rainbow", type=int, default=128, help="(rainbow) hidden width.")
    args = parser.parse_args()

    # --- load split data ---
    csv_path = f"data/processed/{args.ticker}_{args.split}.csv"
    eval_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"{args.split.upper()} set: {len(eval_df)} days of {args.ticker}")

    # --- feature builder ---
    fb = (
        RawOHLCV(window_size=args.window_size)
        if args.features == "raw"
        else OHLCVWithIndicators(window_size=args.window_size)
    )

    # --- environment (full-length episode) ---
    env = TradingEnv(
        df=eval_df,
        feature_builder=fb,
        window_size=args.window_size,
        reward_scheme=args.reward,
        max_episode_steps=None,
    )
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    # --- load agent ---
    if args.agent == "rainbow":
        config = {
            "n_atoms": args.n_atoms,
            "v_min": args.v_min,
            "v_max": args.v_max,
            "hidden": args.hidden_rainbow,
        }
        agent: BaseAgent = RainbowAgent(obs_dim, act_dim, config)
        agent.load(Path(args.checkpoint))
        variant = "Rainbow DQN"
    else:
        config = {
            "hidden": (128, 64, 32),
            "double_dqn": args.double_dqn,
            "epsilon_start": 0.0,
            "epsilon_min": 0.0,
        }
        agent = DQNAgent(obs_dim, act_dim, config)
        agent.load(Path(args.checkpoint))
        variant = "Double DQN" if args.double_dqn else "DQN"
    print(f"Loaded {variant} checkpoint from {args.checkpoint}\n")

    # --- evaluate agent ---
    print("=" * 60)
    print(f"{variant.upper()} PERFORMANCE ({args.split.upper()} SET)")
    print("=" * 60)
    results = evaluate(env, agent)
    for r in results:
        print(f"  Final Value:       ${r['final_value']:,.2f}")
        print(f"  Cumulative Return: {r['cumulative_return']:+.2%}")
        print(f"  Sharpe Ratio:      {r['sharpe_ratio']:.4f}")
        print(f"  Sortino Ratio:     {r['sortino_ratio']:.4f}")
        print(f"  Max Drawdown:      {r['max_drawdown']:.2%}")
        print(f"  Total Trades:      {r['total_trades']}")

    # --- buy & hold benchmark ---
    print("\n" + "=" * 60)
    print(f"BUY & HOLD BASELINE ({args.split.upper()} SET)")
    print("=" * 60)
    bh = buy_and_hold_baseline(env)
    print(f"  Final Value:       ${bh['final_value']:,.2f}")
    print(f"  Cumulative Return: {bh['cumulative_return']:+.2%}")

    # --- comparison ---
    agent_ret = results[0]["cumulative_return"]
    bh_ret = bh["cumulative_return"]
    diff = agent_ret - bh_ret
    print("\n" + "=" * 60)
    print(f"Agent vs Buy & Hold: {diff:+.2%}")
    print("Agent OUTPERFORMS Buy & Hold" if diff > 0 else "Agent UNDERPERFORMS Buy & Hold")
    print("=" * 60)

    if args.no_plot:
        raise SystemExit(0)

    # --- re-run agent to collect trade log for plotting ---
    agent.epsilon = 0.0
    obs, info = env.reset()
    done = False
    while not done:
        mask = info.get("action_mask")
        action = agent.select_action(obs, explore=False, action_mask=mask)
        obs, _, done, _, info = env.step(action)

    from evaluation.plots import plot_behavior

    prices = env.df["Close"].values
    trade_log = info["trade_log"]
    buy_steps = [t["step"] for t in trade_log if t["side"] == "BUY"]
    sell_steps = [t["step"] for t in trade_log if t["side"] == "SELL"]
    profit = results[0]["final_value"] - results[0]["initial_value"]

    plot_behavior(
        prices=prices,
        states_buy=buy_steps,
        states_sell=sell_steps,
        profit=profit,
    )
