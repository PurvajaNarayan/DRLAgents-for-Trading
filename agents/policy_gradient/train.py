"""Training loop for PPO."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from agents.policy_gradient.ppo import PPOAgent
from envs.trading import TradingEnv
from features import OHLCVWithIndicators, RawOHLCV


def train_ppo(
    env: TradingEnv,
    agent: PPOAgent,
    n_episodes: int = 500,
    rollout_steps: int = 2048,
    log_interval: int = 50,
    plot_every: int = 50,
) -> list[dict]:
    """
    Training loop for PPO.

    Collects a fixed-length rollout, then runs multiple epochs
    of minibatch updates on that rollout.
    """
    from evaluation.plots import plot_behavior

    history = []
    obs, info = env.reset()
    ep = 0
    ep_reward = 0.0
    step = 0

    # track buy/sell steps for current episode

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
            profit = pv - env.initial_balance

            if ep % log_interval == 0:
                trades = len(info["trade_log"])
                print(f"Ep {ep:4d} | PV: ${pv:,.2f} | Trades: {trades}")

            # plot behavior after every plot_every episodes
            if ep % plot_every == 0:
                prices = env.df["Close"].values
                trade_log = info["trade_log"]
                buy_steps = [t["step"] for t in trade_log if t["side"] == "BUY"]
                sell_steps = [t["step"] for t in trade_log if t["side"] == "SELL"]
                plot_behavior(
                    prices=prices,
                    states_buy=buy_steps,
                    states_sell=sell_steps,
                    profit=profit,
                    episode=ep,
                )

            history.append(
                {
                    "episode": ep,
                    "portfolio_value": pv,
                    "ep_reward": ep_reward,
                }
            )
            ep_reward = 0.0
            obs, info = env.reset()

        # learn after rollout_steps
        if (step + 1) % rollout_steps == 0 and len(agent.buffer) > 0:
            metrics = agent.learn()
            if ep % log_interval == 0:
                print(
                    f"  PPO update | PL: {metrics['policy_loss']:.4f}"
                    f" | VL: {metrics['value_loss']:.4f}"
                    f" | Ent: {metrics['entropy']:.4f}"
                )

        step += 1

    return history


# ----- CLI entry point -----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--features", choices=["raw", "indicators"], default="raw")
    parser.add_argument(
        "--reward",
        choices=["simple", "sharpe", "sortino", "event_based", "portfolio_delta"],
        default="sharpe",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.05)
    parser.add_argument("--plot_every", type=int, default=50)
    args = parser.parse_args()

    # --- data ---
    train_df = pd.read_csv(f"data/processed/{args.ticker}_train.csv", index_col=0, parse_dates=True)
    print(f"Ticker: {args.ticker} | Training data: {train_df.shape[0]} days")

    # --- feature builder ---
    fb = RawOHLCV(window_size=20) if args.features == "raw" else OHLCVWithIndicators(window_size=20)

    # --- environment ---
    env = TradingEnv(df=train_df, feature_builder=fb, reward_scheme=args.reward, max_episode_steps=252)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"State dim: {obs_dim} | Action dim: {act_dim}")

    # --- agent ---
    config = {
        "gamma": 0.99,
        "lr": args.lr,
        "hidden": 128,
        "clip_eps": 0.2,
        "n_epochs": 10,
        "batch_size": 64,
        "entropy_coef": args.entropy_coef,
    }
    agent = PPOAgent(obs_dim, act_dim, config)

    # --- train ---
    history = train_ppo(
        env,
        agent,
        n_episodes=args.episodes,
        rollout_steps=2048,
        log_interval=50,
        plot_every=args.plot_every,
    )

    # --- save ---
    ckpt_path = Path(f"runs/ppo_{args.ticker}_{args.features}_{args.reward}")
    agent.save(ckpt_path)
    print(f"\nTraining complete! Checkpoint saved to {ckpt_path}")
