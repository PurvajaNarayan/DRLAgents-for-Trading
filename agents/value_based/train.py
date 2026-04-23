"""Training loop for value-based agents (DQN / Double DQN / Rainbow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agents.base import BaseAgent
from agents.value_based.dqn import DQNAgent
from agents.value_based.rainbow import RainbowAgent
from envs.trading import TradingEnv
from features import OHLCVWithIndicators, RawOHLCV


def train_dqn(
    env: TradingEnv,
    agent: BaseAgent,
    n_episodes: int = 500,
    update_every: int = 4,
    log_interval: int = 50,
    plot_every: int = 150,
) -> list[dict]:
    """
    Episodic training loop for value-based agents.

    Works for both DQNAgent and RainbowAgent: the agent exposes a
    `store_transition(...)` method which handles its own buffer
    semantics (uniform replay for DQN, n-step -> prioritized replay
    for Rainbow). Every `update_every` env steps we call `agent.learn()`
    and any per-episode bookkeeping (epsilon decay, n-step flush, ...)
    is delegated to `agent.on_episode_end`.
    """
    from evaluation.plots import plot_behavior

    history: list[dict] = []

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0
        step = 0
        done = False
        losses: list[float] = []

        while not done:
            mask = info.get("action_mask")
            action = agent.select_action(obs, action_mask=mask)
            next_obs, reward, done, _, info = env.step(action)

            next_mask = info.get("action_mask")
            agent.store_transition(obs, action, reward, next_obs, done, next_mask)

            obs = next_obs
            ep_reward += reward
            step += 1

            if step % update_every == 0:
                metrics = agent.learn()
                if metrics["loss"] > 0.0:
                    losses.append(metrics["loss"])

        agent.on_episode_end(ep, info)

        pv = info["portfolio_value"]
        profit = pv - env.initial_balance
        trades = len(info["trade_log"])
        buy_steps = [t["step"] for t in info["trade_log"] if t["side"] == "BUY"]
        sell_steps = [t["step"] for t in info["trade_log"] if t["side"] == "SELL"]

        avg_loss = sum(losses) / len(losses) if losses else 0.0

        if ep % log_interval == 0:
            print(
                f"Ep {ep:4d} | PV: ${pv:,.2f} | Trades: {trades} "
                f"| Eps: {getattr(agent, 'epsilon', 0.0):.3f} | Loss: {avg_loss:.4f} "
                f"| Buffer: {len(agent.buffer)}"
            )

        if ep % plot_every == 0:
            prices = env.df["Close"].values
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
                "epsilon": getattr(agent, "epsilon", 0.0),
                "avg_loss": avg_loss,
                "trades": trades,
            }
        )

    return history


def train_dqn_sliding(
    env: TradingEnv,
    agent: BaseAgent,
    episode_length: int = 180,
    episode_stride: int = 30,
    num_passes: int = 3,
    update_every: int = 4,
    log_interval: int = 10,
    plot_every: int = 500,
    seed: int | None = None,
) -> list[dict]:
    """
    The total number of episodes is NOT specified explicitly; it's derived
    from the data length, `window_size`, `episode_length`, `episode_stride`,
    and `num_passes`:

        episode_starts = [window_size, window_size + stride, ...
                          while (start + episode_length) < len(df)]
        total_episodes = len(episode_starts) * num_passes

    Each episode deterministically starts at one of these offsets. Within a
    pass the order is shuffled (for sample diversity).

    This function does NOT modify `TradingEnv`. It reuses `env.reset()` for
    portfolio/bookkeeping resets, then overrides the episode start index
    before the first action so each trajectory covers the chosen slice.
    """
    from evaluation.plots import plot_behavior

    rng = np.random.default_rng(seed)
    history: list[dict] = []

    data_len = len(env.df)
    window_size = env.window_size
    episode_starts: list[int] = []
    start = window_size
    while start + episode_length < data_len:
        episode_starts.append(start)
        start += episode_stride

    total_episodes = len(episode_starts) * num_passes
    print(
        f"Sliding schedule | data={data_len} days | window={window_size} "
        f"| episode_length={episode_length} | stride={episode_stride} "
        f"| passes={num_passes}"
    )
    print(f"Episodes per pass: {len(episode_starts)} | Total episodes: {total_episodes}")

    # Make the env respect our episode length; restore on exit.
    prev_max_steps = env.max_episode_steps
    env.max_episode_steps = episode_length

    try:
        global_ep = 0
        for pass_num in range(num_passes):
            shuffled = episode_starts.copy()
            rng.shuffle(shuffled)

            for start_idx in shuffled:
                global_ep += 1

                # Normal reset (clears portfolio, returns, trade log, etc.)...
                _, _ = env.reset()
                # ...then override the starting index to our sliding-window slot.
                env._current_step = int(start_idx)
                env._steps_in_episode = 0
                obs = env._get_observation()
                info = env._get_info()

                ep_reward = 0.0
                step = 0
                done = False
                losses: list[float] = []

                while not done:
                    mask = info.get("action_mask")
                    action = agent.select_action(obs, action_mask=mask)
                    next_obs, reward, done, _, info = env.step(action)

                    next_mask = info.get("action_mask")
                    agent.store_transition(obs, action, reward, next_obs, done, next_mask)

                    obs = next_obs
                    ep_reward += reward
                    step += 1

                    if step % update_every == 0:
                        metrics = agent.learn()
                        if metrics["loss"] > 0.0:
                            losses.append(metrics["loss"])

                agent.on_episode_end(global_ep, info)

                pv = info["portfolio_value"]
                profit = pv - env.initial_balance
                trades = len(info["trade_log"])
                buy_steps = [t["step"] for t in info["trade_log"] if t["side"] == "BUY"]
                sell_steps = [t["step"] for t in info["trade_log"] if t["side"] == "SELL"]
                avg_loss = sum(losses) / len(losses) if losses else 0.0

                if global_ep % log_interval == 0:
                    print(
                        f"Pass {pass_num + 1}/{num_passes} "
                        f"| Ep {global_ep:4d}/{total_episodes} "
                        f"| start={start_idx:4d} "
                        f"| PV: ${pv:,.2f} | Trades: {trades} "
                        f"| Eps: {getattr(agent, 'epsilon', 0.0):.3f} | Loss: {avg_loss:.4f} "
                        f"| Buffer: {len(agent.buffer)}"
                    )

                if global_ep % plot_every == 0:
                    prices = env.df["Close"].values
                    plot_behavior(
                        prices=prices,
                        states_buy=buy_steps,
                        states_sell=sell_steps,
                        profit=profit,
                        episode=global_ep,
                    )

                history.append(
                    {
                        "episode": global_ep,
                        "pass": pass_num + 1,
                        "start_idx": int(start_idx),
                        "portfolio_value": pv,
                        "ep_reward": ep_reward,
                        "epsilon": getattr(agent, "epsilon", 0.0),
                        "avg_loss": avg_loss,
                        "trades": trades,
                    }
                )
    finally:
        env.max_episode_steps = prev_max_steps

    return history


# ----- CLI entry point -----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train value-based agent (DQN / Double DQN / Rainbow)"
    )
    parser.add_argument(
        "--agent",
        choices=["dqn", "rainbow"],
        default="dqn",
        help="Which value-based agent to train.",
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--features", choices=["raw", "indicators"], default="indicators")
    parser.add_argument(
        "--reward",
        choices=["simple", "sharpe", "sortino", "event_based", "portfolio_delta"],
        default="portfolio_delta",
    )
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=50_000,
        help="Max replay transitions (FIFO for DQN; PER for Rainbow).",
    )
    parser.add_argument("--target_update_freq", type=int, default=500)
    parser.add_argument("--train_start", type=int, default=500)
    parser.add_argument("--epsilon_start", type=float, default=0.3)
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument(
        "--double_dqn",
        action="store_true",
        help="(dqn) Use Double DQN target.",
    )
    parser.add_argument(
        "--loss",
        choices=["huber", "mse"],
        default="mse",
        help="(dqn) huber is more stable for large dollar rewards.",
    )
    parser.add_argument(
        "--update_every",
        type=int,
        default=4,
        help="Env steps between each learn() call.",
    )
    parser.add_argument("--max_episode_steps", type=int, default=252)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--plot_every", type=int, default=1_000_000)
    parser.add_argument("--log_interval", type=int, default=10)

    # --- episode schedule ---
    parser.add_argument(
        "--schedule",
        choices=["random", "sliding"],
        default="random",
        help=(
            "'random': fixed n_episodes of random-start rollouts. "
            "'sliding': deterministic sliding-window episodes."
        ),
    )
    parser.add_argument("--episode_length", type=int, default=180)
    parser.add_argument("--episode_stride", type=int, default=30)
    parser.add_argument("--num_passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)

    # --- Rainbow-only hyperparameters ---
    rb = parser.add_argument_group("rainbow")
    rb.add_argument("--n_atoms", type=int, default=51)
    rb.add_argument("--v_min", type=float, default=-10.0)
    rb.add_argument("--v_max", type=float, default=10.0)
    rb.add_argument("--n_step", type=int, default=3)
    rb.add_argument("--per_alpha", type=float, default=0.5)
    rb.add_argument("--per_beta_start", type=float, default=0.4)
    rb.add_argument("--per_beta_end", type=float, default=1.0)
    rb.add_argument("--per_beta_steps", type=int, default=100_000)
    rb.add_argument("--noisy_sigma", type=float, default=0.5)
    rb.add_argument("--hidden_rainbow", type=int, default=64)

    args = parser.parse_args()

    # --- data ---
    train_df = pd.read_csv(
        f"data/processed/{args.ticker}_train.csv", index_col=0, parse_dates=True
    )
    print(f"Ticker: {args.ticker} | Training data: {train_df.shape[0]} days")

    # --- feature builder ---
    fb = (
        RawOHLCV(window_size=args.window_size)
        if args.features == "raw"
        else OHLCVWithIndicators(window_size=args.window_size)
    )

    # --- environment ---
    env = TradingEnv(
        df=train_df,
        feature_builder=fb,
        window_size=args.window_size,
        reward_scheme=args.reward,
        max_episode_steps=args.max_episode_steps,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"State dim: {obs_dim} | Action dim: {act_dim}")

    # --- agent ---
    if args.agent == "rainbow":
        config = {
            "gamma": args.gamma,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "target_update_freq": args.target_update_freq,
            "train_start": args.train_start,
            "n_atoms": args.n_atoms,
            "v_min": args.v_min,
            "v_max": args.v_max,
            "n_step": args.n_step,
            "per_alpha": args.per_alpha,
            "per_beta_start": args.per_beta_start,
            "per_beta_end": args.per_beta_end,
            "per_beta_steps": args.per_beta_steps,
            "noisy_sigma": args.noisy_sigma,
            "hidden": args.hidden_rainbow,
        }
        agent: BaseAgent = RainbowAgent(obs_dim, act_dim, config)
        variant = "rainbow"
    else:
        config = {
            "gamma": args.gamma,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "target_update_freq": args.target_update_freq,
            "train_start": args.train_start,
            "epsilon_start": args.epsilon_start,
            "epsilon_min": args.epsilon_min,
            "epsilon_decay": args.epsilon_decay,
            "double_dqn": args.double_dqn,
            "loss": args.loss,
            "hidden": (128, 64, 32),
        }
        agent = DQNAgent(obs_dim, act_dim, config)
        variant = "double_dqn" if args.double_dqn else "dqn"

    print(f"Variant: {variant} | Schedule: {args.schedule}")

    # --- train ---
    if args.schedule == "sliding":
        history = train_dqn_sliding(
            env,
            agent,
            episode_length=args.episode_length,
            episode_stride=args.episode_stride,
            num_passes=args.num_passes,
            update_every=args.update_every,
            log_interval=args.log_interval,
            plot_every=args.plot_every,
            seed=args.seed,
        )
    else:
        history = train_dqn(
            env,
            agent,
            n_episodes=args.episodes,
            update_every=args.update_every,
            log_interval=args.log_interval,
            plot_every=args.plot_every,
        )

    # --- save ---
    ckpt_path = Path(f"runs/{variant}_{args.ticker}_{args.features}_{args.reward}")
    agent.save(ckpt_path)
    print(f"\nTraining complete! Checkpoint saved to {ckpt_path}")
