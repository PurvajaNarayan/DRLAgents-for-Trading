"""Evaluate PPO and PMDP on both Validation and Test sets, then plot comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from envs.trading import TradingEnv
from evaluation.policy_gradient.evaluate import buy_and_hold_baseline
from features import OHLCVWithIndicators, RawOHLCV


def run_evaluation(df, env_cls, fb, reward, agent_type, agent):
    env = env_cls(df=df, feature_builder=fb, reward_scheme=reward)

    if agent_type == "pmdp":
        from evaluation.partial_mdp.evaluate import evaluate as eval_fn
    else:
        from evaluation.policy_gradient.evaluate import evaluate as eval_fn

    results = eval_fn(env, agent)
    agent_ret = results[0]["cumulative_return"]

    bh = buy_and_hold_baseline(env)
    bh_ret = bh["cumulative_return"]

    return agent_ret, bh_ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PPO and PMDP vs baselines")
    parser.add_argument("--ticker", default="AMZN")
    parser.add_argument("--features", choices=["raw", "indicators"], default="indicators")
    parser.add_argument("--reward", default="event_based")
    args = parser.parse_args()

    # --- Data ---
    val_df = pd.read_csv(f"data/processed/{args.ticker}_val.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv(f"data/processed/{args.ticker}_test.csv", index_col=0, parse_dates=True)

    # --- Feature Builder ---
    fb = RawOHLCV(window_size=20) if args.features == "raw" else OHLCVWithIndicators(window_size=20)

    # --- Get dims ---
    env_tmp = TradingEnv(df=val_df, feature_builder=fb, reward_scheme=args.reward)
    obs_dim = int(env_tmp.observation_space.shape[0])
    act_dim = int(env_tmp.action_space.n)

    # --- Load PPO ---
    from agents.policy_gradient.ppo import PPOAgent
    ppo_config = {"gamma": 0.99, "lr": 3e-4, "hidden": 128, "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64, "entropy_coef": 0.05}
    ppo_agent = PPOAgent(obs_dim, act_dim, ppo_config)
    ppo_agent.load(Path(f"runs/ppo_{args.ticker}_{args.features}_{args.reward}"))
    print("Loaded PPO")

    # --- Load PMDP ---
    from agents.partial_mdp.agent import PMDPAgent
    pmdp_config = {"gamma": 0.99, "lr": 3e-4, "hidden": 128, "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64}
    pmdp_agent = PMDPAgent(obs_dim, act_dim, pmdp_config)
    pmdp_agent.load(Path(f"runs/pmdp_{args.ticker}_{args.features}_{args.reward}"))
    print("Loaded PMDP")

    # --- Validation ---
    print("\n--- Evaluating on Validation Set ---")
    ppo_val_ret, bh_val_ret = run_evaluation(val_df, TradingEnv, fb, args.reward, "ppo", ppo_agent)
    pmdp_val_ret, _ = run_evaluation(val_df, TradingEnv, fb, args.reward, "pmdp", pmdp_agent)

    # --- Test ---
    print("\n--- Evaluating on Test Set ---")
    ppo_test_ret, bh_test_ret = run_evaluation(test_df, TradingEnv, fb, args.reward, "ppo", ppo_agent)
    pmdp_test_ret, _ = run_evaluation(test_df, TradingEnv, fb, args.reward, "pmdp", pmdp_agent)

    # --- DQN results for AMZN (hardcoded from Prashanth's evaluation) ---
    dqn_val_ret = 0.0066  # +0.66%
    dqn_test_ret = 0.1885  # +18.88%

    # # --- DQN results for CHWY (hardcoded from Prashanth's evaluation) ---
    # dqn_val_ret = 1.6448   # +164.48%
    # dqn_test_ret = 0.1988  # +19.88%

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"{'':12s} | {'Validation':>12s} | {'Test':>12s}")
    print(f"{'DQN':12s} | {dqn_val_ret:+12.2%} | {dqn_test_ret:+12.2%}")
    print(f"{'PPO':12s} | {ppo_val_ret:+12.2%} | {ppo_test_ret:+12.2%}")
    print(f"{'PMDP':12s} | {pmdp_val_ret:+12.2%} | {pmdp_test_ret:+12.2%}")
    print(f"{'Buy & Hold':12s} | {bh_val_ret:+12.2%} | {bh_test_ret:+12.2%}")

    # --- Plots ---
    from evaluation.plots import plot_val_vs_baseline, plot_test_vs_baseline

    save_dir = Path(f"runs/comparison_{args.ticker}_{args.features}_{args.reward}")
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_val_vs_baseline(
        val_metrics={"DQN": dqn_val_ret, "PPO": ppo_val_ret, "PMDP": pmdp_val_ret},
        baseline_value=bh_val_ret,
        ticker=args.ticker,
        save_path=str(save_dir / "val_vs_baseline.png"),
    )

    plot_test_vs_baseline(
        test_metrics={"DQN": dqn_test_ret, "PPO": ppo_test_ret, "PMDP": pmdp_test_ret},
        baseline_value=bh_test_ret,
        ticker=args.ticker,
        save_path=str(save_dir / "test_vs_baseline.png"),
    )