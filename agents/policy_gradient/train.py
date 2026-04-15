"""Training loops for REINFORCE, REINFORCE+Baseline, and PPO."""

from __future__ import annotations

import pandas as pd

from agents.policy_gradient.ppo import PPOAgent
from agents.policy_gradient.reinforce import ReinforceAgent
from agents.policy_gradient.reinforce_baseline import ReinforceBaselineAgent
from envs.trading import TradingEnv
from features import OHLCVWithIndicators, RawOHLCV


def train_reinforce(env: TradingEnv, agent: ReinforceAgent, n_episodes: int = 500):
    """
    Training loop for REINFORCE and REINFORCE+Baseline.

    Both use the same episode-based loop:
      1. Collect full episode
      2. Call learn() once
    """
    history = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            mask = info.get("action_mask")
            action = agent.select_action(obs, action_mask=mask)
            obs, reward, done, _, info = env.step(action)
            agent.store_reward(reward)
            ep_reward += reward

        metrics = agent.learn()
        metrics["episode"] = ep
        metrics["portfolio_value"] = info["portfolio_value"]
        metrics["ep_reward"] = ep_reward
        history.append(metrics)

        if ep % 50 == 0:
            pv = info["portfolio_value"]
            trades = len(info["trade_log"])
            print(
                f"Ep {ep:4d} | PV: ${pv:,.2f} | Trades: {trades} | Loss: {metrics.get('loss', metrics.get('policy_loss', 0)):.4f}"
            )

    return history


def train_ppo(
    env: TradingEnv,
    agent: PPOAgent,
    n_episodes: int = 500,
    rollout_steps: int = 2048,
):
    """
    Training loop for PPO.

    PPO collects a fixed-length rollout, then runs multiple epochs
    of minibatch updates on that rollout.
    """
    history = []
    obs, info = env.reset()
    ep = 0
    ep_reward = 0.0
    step = 0

    while ep < n_episodes:
        mask = info.get("action_mask")
        action, log_prob, value = agent.select_action(obs, action_mask=mask)
        next_obs, reward, done, _, info = env.step(action)

        agent.buffer.store(
            obs=obs,
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
            action_mask=mask,
        )

        obs = next_obs
        ep_reward += reward

        if done:
            ep += 1
            pv = info["portfolio_value"]

            if ep % 50 == 0:
                trades = len(info["trade_log"])
                print(f"Ep {ep:4d} | PV: ${pv:,.2f} | Trades: {trades}")

            history.append(
                {
                    "episode": ep,
                    "portfolio_value": pv,
                    "ep_reward": ep_reward,
                }
            )
            ep_reward = 0.0
            obs, info = env.reset()

        # --- learn after rollout_steps ---
        if (step + 1) % rollout_steps == 0 and len(agent.buffer) > 0:
            metrics = agent.learn()
            if ep % 50 == 0:
                print(
                    f"  PPO update | PL: {metrics['policy_loss']:.4f} | VL: {metrics['value_loss']:.4f} | Ent: {metrics['entropy']:.4f}"
                )
                
        step += 1

    return history


# ----- CLI entry point -----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train policy gradient agents")
    parser.add_argument("--agent", choices=["reinforce", "baseline", "ppo"], default="ppo")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--features", choices=["raw", "indicators"], default="raw")
    parser.add_argument("--reward", choices=["simple", "sharpe", "sortino", "action_simple", "action_sharpe", "action_sortino"], default="sharpe")
    parser.add_argument("--max_episode_steps", type=int, default=252, help="Max steps per episode (default: 252 for 1 year)")
    args = parser.parse_args()

    # --- load data ---
    train_df = pd.read_csv(
        f"data/processed/{args.ticker}_train.csv", index_col=0, parse_dates=["Date"]
    )

    # --- feature builder ---
    fb = RawOHLCV(window_size=20) if args.features == "raw" else OHLCVWithIndicators(window_size=10)

    # --- environment ---
    print("Size of the training dataset is:",train_df.shape)
    env = TradingEnv(df=train_df, feature_builder=fb, reward_scheme=args.reward, max_episode_steps=args.max_episode_steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # --- config ---
    config = {"gamma": 0.99, "lr": 3e-4, "hidden": 128}

    # --- agent ---
    if args.agent == "reinforce":
        agent = ReinforceAgent(obs_dim, act_dim, config)
        train_reinforce(env, agent, args.episodes)
    elif args.agent == "baseline":
        agent = ReinforceBaselineAgent(obs_dim, act_dim, config)
        train_reinforce(env, agent, args.episodes)  # same loop, different agent
    elif args.agent == "ppo":
        config.update({"clip_eps": 0.2, "n_epochs": 10, "batch_size": 64})
        agent = PPOAgent(obs_dim, act_dim, config)
        train_ppo(env, agent, args.episodes, rollout_steps=64)

    # --- save checkpoint ---
    from pathlib import Path

    ckpt_path = Path(f"runs/{args.agent}_{args.ticker}_{args.features}")
    agent.save(ckpt_path)
    print(f"\nTraining complete! Checkpoint saved to {ckpt_path}")
