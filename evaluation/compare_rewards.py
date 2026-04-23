"""Compare cumulative rewards across agents and reward functions on the test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from envs.trading import TradingEnv
from evaluation.plots import plot_reward_comparison
from features import OHLCVWithIndicators, RawOHLCV


# Maps agent name -> (checkpoint prefix, model file, loader function)
AGENT_CONFIG = {
    "ppo": {
        "prefix": "ppo",
        "model_file": "ppo.pt",
    },
    "dqn": {
        "prefix": "double_dqn",
        "model_file": "dqn.pt",
    },
    "pomdp": {
        "prefix": "pmdp",
        "model_file": "pmdp.pt",
    },
}


def load_agent(agent_type: str, obs_dim: int, act_dim: int, ckpt_path: Path):
    """Instantiate and load the correct agent class from checkpoint."""
    if agent_type == "ppo":
        from agents.policy_gradient.ppo import PPOAgent
        config = {
            "gamma": 0.99, "lr": 3e-4, "hidden": 128,
            "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64,
        }
        agent = PPOAgent(obs_dim, act_dim, config)
        agent.load(ckpt_path)
        return agent

    elif agent_type == "dqn":
        from agents.value_based.dqn import DQNAgent
        config = {
            "gamma": 0.99, "lr": 5e-4, "hidden": (128, 64, 32),
            "double_dqn": True, "epsilon_start": 0.0, "epsilon_min": 0.0,
        }
        agent = DQNAgent(obs_dim, act_dim, config)
        agent.load(ckpt_path)
        return agent

    elif agent_type == "pomdp":
        from agents.partial_mdp.agent import PMDPAgent
        config = {
            "gamma": 0.99, "lr": 3e-4, "lr_lstm": 1e-3, "hidden": 128,
            "clip_eps": 0.2, "n_epochs": 10, "batch_size": 64,
        }
        agent = PMDPAgent(obs_dim, act_dim, config)
        agent.load(ckpt_path)
        return agent

    raise ValueError(f"Unknown agent type: {agent_type}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare reward functions across agents")
    parser.add_argument("--agent", choices=["ppo", "dqn", "pomdp"], default="ppo")
    parser.add_argument("--ticker", default="CHWY")
    parser.add_argument("--features", choices=["raw", "indicators"], default="indicators")
    parser.add_argument(
        "--split", choices=["val", "test"], default="test",
        help="Data split to evaluate on."
    )
    parser.add_argument(
        "--rewards",
        nargs="+",
        default=None,
        help="Reward schemes to compare. Defaults to all available checkpoints."
    )
    parser.add_argument("--window_size", type=int, default=20)
    args = parser.parse_args()

    cfg = AGENT_CONFIG[args.agent]
    prefix = cfg["prefix"]
    model_file = cfg["model_file"]

    # --- load split data ---
    csv_path = f"data/processed/{args.ticker}_{args.split}.csv"
    eval_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"{args.split.upper()} set: {len(eval_df)} days of {args.ticker}\n")

    fb = (
        RawOHLCV(window_size=args.window_size)
        if args.features == "raw"
        else OHLCVWithIndicators(window_size=args.window_size)
    )

    # --- auto-discover checkpoints if --rewards not specified ---
    runs_dir = Path("runs")
    if args.rewards:
        reward_schemes = args.rewards
    else:
        pattern = f"{prefix}_{args.ticker}_{args.features}_*"
        matched = sorted(runs_dir.glob(pattern))
        reward_schemes = []
        for p in matched:
            if (p / model_file).exists():
                # extract reward name from folder suffix
                suffix = p.name.replace(f"{prefix}_{args.ticker}_{args.features}_", "")
                reward_schemes.append(suffix)

    if not reward_schemes:
        print(f"No checkpoints found matching: runs/{prefix}_{args.ticker}_{args.features}_*/{model_file}")
        print("Train first, e.g.:")
        print(f"  python -m agents.policy_gradient.train --ticker {args.ticker} --reward event_based")
        raise SystemExit(1)

    print(f"Agent:   {args.agent.upper()}")
    print(f"Ticker:  {args.ticker}")
    print(f"Split:   {args.split}")
    print(f"Schemes: {reward_schemes}\n")

    # --- collect rewards per scheme ---
    all_rewards: dict[str, list[float]] = {}

    for scheme in reward_schemes:
        ckpt_path = runs_dir / f"{prefix}_{args.ticker}_{args.features}_{scheme}"

        if not (ckpt_path / model_file).exists():
            print(f"  [{scheme}] checkpoint not found at {ckpt_path} — skipping")
            continue

        env = TradingEnv(
            df=eval_df,
            feature_builder=fb,
            window_size=args.window_size,
            reward_scheme=scheme,
            max_episode_steps=None,
        )
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.n)

        try:
            agent = load_agent(args.agent, obs_dim, act_dim, ckpt_path)
        except Exception as e:
            print(f"  [{scheme}] failed to load checkpoint: {e} — skipping")
            continue

        rewards = collect_rewards(env, agent, args.agent)
        all_rewards[scheme.replace("_", " ").title()] = rewards
        print(
            f"  [{scheme:>15s}] steps={len(rewards):4d} | "
            f"total reward={sum(rewards):+.4f}"
        )

    # --- plot ---
    if not all_rewards:
        print("\nNo valid checkpoints found. Nothing to plot.")
        raise SystemExit(1)

    save_dir = Path(f"runs/{prefix}_{args.ticker}_{args.features}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / f"reward_comparison_{args.split}.png")

    plot_reward_comparison(
        results=all_rewards,
        title=f"Cumulative Reward — {args.agent.upper()} on {args.ticker} ({args.split.title()} Set)",
        save_path=save_path,
    )
