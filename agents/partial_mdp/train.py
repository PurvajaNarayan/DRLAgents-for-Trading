"""Training loop for PMDP Actor-Critic."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from agents.partial_mdp.agent import PMDPAgent
from envs.trading import TradingEnv
from features import OHLCVWithIndicators, RawOHLCV


def pretrain_lstm(env: TradingEnv, agent: PMDPAgent, n_episodes: int = 50) -> None:
    """Phase 1: Pretrain the LSTM state predictor using random exploration."""
    print("--- Phase 1: Pretraining LSTM Predictor ---")
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        obs_list = [obs]
        act_list = []
        
        while not done:
            mask = info.get("action_mask")
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)
            
            obs, _, done, _, info = env.step(action)
            
            obs_list.append(obs)
            
        loss = agent.learn_predictor(obs_list)
        if (ep + 1) % 10 == 0:
            print(f"Pretrain LSTM Ep {ep + 1:3d} | MSE Loss: {loss:.6f}")


def train_pmdp_ppo(
    env: TradingEnv, agent: PMDPAgent, n_episodes: int = 500, rollout_steps: int = 2048
) -> list[dict]:
    """Phase 2: Train the Actor-Critic using the frozen LSTM state predictor."""
    print("\n--- Phase 2: Training PMDP Actor-Critic ---")
    
    agent.freeze_predictor()
    
    history = []
    obs, info = env.reset()
    agent.reset_hidden_state()
    ep = 0
    ep_reward = 0.0
    step = 0

    while ep < n_episodes:
        mask = info.get("action_mask")
        
        # Get action from PPO using LSTM context
        hx = agent.current_h
        action, log_prob, value = agent.select_action(obs, action_mask=mask)
        
        # Step env
        next_obs, reward, done, _, info = env.step(action)

        # Store to buffer WITH hx
        agent.buffer.store(
            obs=obs,
            hx=hx,
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
            action_mask=mask,
        )

        # Step LSTM forward
        agent.update_hidden_state(obs)

        obs = next_obs
        ep_reward += reward

        if done:
            ep += 1
            pv = info["portfolio_value"]

            if ep % 50 == 0:
                trades = len(info["trade_log"])
                print(f"Ep {ep:4d} | PV: ${pv:,.2f} | Trades: {trades}")

            history.append({
                "episode": ep,
                "portfolio_value": pv,
                "ep_reward": ep_reward,
            })
            ep_reward = 0.0
            
            obs, info = env.reset()
            agent.reset_hidden_state()

        # Learn after gathering rollout
        if (step + 1) % rollout_steps == 0 and len(agent.buffer) > 0:
            metrics = agent.learn()
            if ep % 50 == 0:
                print(f"  PPO update | PL: {metrics['policy_loss']:.4f} | VL: {metrics['value_loss']:.4f}")
                
        step += 1

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PMDP Actor-Critic")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--pretrain_eps", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--features", choices=["raw", "indicators"], default="raw")
    parser.add_argument("--reward", choices=["simple", "sharpe", "sortino"], default="sharpe")
    args = parser.parse_args()

    train_df = pd.read_csv(f"data/processed/{args.ticker}_train.csv", index_col=0, parse_dates=["Date"])
    fb = RawOHLCV(window_size=20) if args.features == "raw" else OHLCVWithIndicators(window_size=10)

    print("Size of the training dataset is:", train_df.shape)
    env = TradingEnv(df=train_df, feature_builder=fb, reward_scheme=args.reward)
    
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

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

    # Phase 1: Train Predictor
    pretrain_lstm(env, agent, args.pretrain_eps)
    
    # Phase 2: Train Actor-Critic
    train_pmdp_ppo(env, agent, args.episodes)

    # Save
    ckpt_path = Path(f"runs/pmdp_{args.ticker}_{args.features}")
    agent.save(ckpt_path)
    print(f"\nTraining complete! Checkpoint saved to {ckpt_path}")
