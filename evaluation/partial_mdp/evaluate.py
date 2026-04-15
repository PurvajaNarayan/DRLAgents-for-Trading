"""Evaluate a trained PMDP Actor-Critic agent on the test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agents.partial_mdp.agent import PMDPAgent
from envs.trading import TradingEnv
from features import OHLCVWithIndicators, RawOHLCV
from evaluation.policy_gradient.evaluate import buy_and_hold_baseline


def evaluate(env: TradingEnv, agent: PMDPAgent, n_episodes: int = 1) -> list[dict]:
    """
    Run the PMDP agent on the test set with NO exploration (greedy actions).
    Provides consecutive step updates to the inner LSTM.
    Returns metrics for each episode.
    """
    results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset_hidden_state()
        
        done = False
        ep_reward = 0.0
        daily_returns = []

        initial_value = info["portfolio_value"]

        while not done:
            mask = info.get("action_mask")

            # --- greedy action (explore=False) ---
            action, _, _ = agent.select_action(obs, explore=False, action_mask=mask)

            prev_pv = info["portfolio_value"]
            next_obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            
            # --- update LSTM State ---
            agent.update_hidden_state(obs)
            obs = next_obs

            # track daily return
            curr_pv = info["portfolio_value"]
            daily_ret = (curr_pv - prev_pv) / prev_pv if prev_pv > 0 else 0.0
            daily_returns.append(daily_ret)

        # --- compute metrics ---
        final_value = info["portfolio_value"]
        returns_arr = np.array(daily_returns)

        cumulative_return = (final_value - initial_value) / initial_value
        sharpe = (
            np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)
            if np.std(returns_arr) > 1e-8
            else 0.0
        )

        # sortino (annualized)
        downside = returns_arr[returns_arr < 0]
        sortino = (
            np.mean(returns_arr) / np.std(downside) * np.sqrt(252)
            if len(downside) > 0 and np.std(downside) > 1e-8
            else 0.0
        )

        # max drawdown
        portfolio_values = np.array(env._portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)

        # win rate
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

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PMDP agent on test set")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--features", choices=["raw", "indicators"], default="raw")
    parser.add_argument("--reward", choices=["simple", "sharpe", "sortino", "action_simple", "action_sharpe", "action_sortino"], default="sharpe")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder")
    args = parser.parse_args()

    # --- load TEST data ---
    test_df = pd.read_csv(
        f"data/processed/{args.ticker}_test.csv", index_col=0, parse_dates=["Date"]
    )
    print(f"Test set: {len(test_df)} days of {args.ticker}")

    # --- feature builder ---
    fb = RawOHLCV(window_size=20) if args.features == "raw" else OHLCVWithIndicators(window_size=10)

    # --- environment ---
    env = TradingEnv(df=test_df, feature_builder=fb, reward_scheme=args.reward, max_episode_steps=None)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    # --- debug: print full state at step 20 ---
    obs, info = env.reset()  # starts at step = window_size (20)
    print("=" * 60)
    print(f"STATE REPRESENTATION AT STEP {info['step']}")
    print(f"Feature builder: {type(fb).__name__}")
    print(f"State dimension: {len(obs)}")
    print("=" * 60)

    if isinstance(fb, OHLCVWithIndicators):
        ohlcv_size = fb.window_size * 5
        ohlcv_part = obs[:ohlcv_size].reshape(fb.window_size, 5)
        indicators = obs[ohlcv_size:-1]
        position = obs[-1]

        # get actual dates from the dataframe
        step = info["step"]
        dates = env.df.index[step - fb.window_size : step]

        print("\n--- OHLCV Window (20 days x 5, normalized 0-1) ---")
        print(
            f"{'Day':>4} {'Date':>12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>8}"
        )
        for i, (date, row) in enumerate(zip(dates, ohlcv_part)):
            date_str = str(date)[:10]
            print(
                f"{i + 1:4d} {date_str:>12} {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f} {row[4]:8.4f}"
            )

        print(f"\n--- Technical Indicators (from {str(dates[-1])[:10]}) ---")
        labels = [
            "RSI (0-1)",
            "MACD (rel)",
            "MACD Signal (rel)",
            "BB Upper Dist",
            "BB Lower Dist",
            "ATR (rel)",
        ]
        for label, val in zip(labels, indicators):
            print(f"  {label:<20s}: {val:+.6f}")

        print("\n--- Position Flag ---")
        print(f"  Holding shares: {'Yes' if position == 1.0 else 'No'}")
        print(f"  Trading date:    {str(env.df.index[step])[:10]}")

    else:
        ohlcv_size = fb.window_size * 5
        ohlcv_part = obs[:ohlcv_size].reshape(fb.window_size, 5)
        position = obs[-1]

        step = info["step"]
        dates = env.df.index[step - fb.window_size : step]

        print("\n--- OHLCV Window (20 days x 5, normalized 0-1) ---")
        print(
            f"{'Day':>4} {'Date':>12} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>8}"
        )
        for i, (date, row) in enumerate(zip(dates, ohlcv_part)):
            date_str = str(date)[:10]
            print(
                f"{i + 1:4d} {date_str:>12} {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f} {row[4]:8.4f}"
            )

        print("\n--- Position Flag ---")
        print(f"  Holding shares: {'Yes' if position == 1.0 else 'No'}")
        print(f"  Trading date:    {str(env.df.index[step])[:10]}")

    print("\n--- Raw Vector ---")
    print(obs)
    print("=" * 60 + "\n")

    # --- load agent ---
    config = {
        "gamma": 0.99,
        "lr": 3e-4,
        "lr_lstm": 1e-3,
        "hidden": 128,
        "clip_eps": 0.2,
        "n_epochs": 10,
        "batch_size": 64
    }

    agent = PMDPAgent(obs_dim, act_dim, config)

    agent.load(Path(args.checkpoint))
    # Note: freeze predictor just in case to ensure we don't accidentally run train blocks (though torch no_grad on eval makes it safe anyway)
    agent.freeze_predictor()
    print(f"Loaded checkpoint from {args.checkpoint}\n")

    # --- evaluate agent ---
    print("=" * 60)
    print("AGENT PERFORMANCE (Test Set)")
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
    print("BUY & HOLD BASELINE (Test Set)")
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
    if diff > 0:
        print("Agent OUTPERFORMS Buy & Hold")
    else:
        print("Agent UNDERPERFORMS Buy & Hold")
    print("=" * 60)
