# Value-Based Agents (DQN, Double DQN, Rainbow)

This module implements value-based reinforcement learning agents for trading:

- **DQN**: Deep Q-Network with experience replay and target networks
- **Double DQN**: Mitigates overestimation bias by decoupling action selection from evaluation
- **Rainbow DQN**: Combines 6 improvements: Double DQN + Dueling Networks + Prioritized Experience Replay + Multi-step Learning + Distributional RL (C51) + Noisy Networks

All agents work with the `TradingEnv` and support action masking for valid trading actions (buy/sell/hold constraints).

## Files Structure

```
agents/value_based/
├── __init__.py          # Public exports
├── network.py           # QNetwork, NoisyLinear, RainbowNetwork architectures  
├── replay.py            # ReplayBuffer, PrioritizedReplayBuffer, NStepBuffer
├── dqn.py              # DQNAgent implementation
├── rainbow.py          # RainbowAgent implementation
├── train.py            # Training loops and CLI
└── README.md           # This file
```

## Quick Start

### 1. Train a Double DQN Agent

```bash
python -m agents.value_based.train \
  --agent dqn --double_dqn \
  --ticker AAPL \
  --features indicators \
  --reward portfolio_delta \
  --window_size 30 \
  --episodes 500 \
  --lr 5e-4 --batch_size 64 --buffer_size 50000 \
  --epsilon_start 0.3 --epsilon_decay 0.995 \
  --update_every 4
```

**Output**: Saves checkpoint to `runs/double_dqn_AAPL_indicators_portfolio_delta/dqn.pt`

### 2. Train a Rainbow DQN Agent

```bash
python -m agents.value_based.train \
  --agent rainbow \
  --ticker AAPL \
  --features indicators \
  --reward portfolio_delta \
  --window_size 30 \
  --episodes 500 \
  --lr 1e-4 --batch_size 64 --buffer_size 50000 \
  --train_start 1000 --target_update_freq 500 --update_every 4 \
  --n_atoms 51 --v_min -500 --v_max 500 \
  --n_step 3 \
  --per_alpha 0.5 --per_beta_start 0.4 --per_beta_end 1.0 \
  --noisy_sigma 0.5 --hidden_rainbow 128
```

**Output**: Saves checkpoint to `runs/rainbow_AAPL_indicators_portfolio_delta/rainbow.pt`

## Training Options

### Core Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--agent` | `dqn` | Agent type: `dqn` or `rainbow` |
| `--ticker` | `AAPL` | Stock ticker symbol |
| `--features` | `indicators` | Feature type: `raw` (OHLCV only) or `indicators` (OHLCV + technical indicators) |
| `--reward` | `portfolio_delta` | Reward scheme: `simple`, `sharpe`, `sortino`, `event_based`, `portfolio_delta` |
| `--window_size` | `30` | Lookback window for features |
| `--episodes` | `500` | Number of training episodes (for random schedule) |

### Episode Schedules

**Random Episodes** (default):
```bash
--schedule random --episodes 500 --max_episode_steps 252
```
Each episode starts at a random point in the training data.

**Sliding Window** (notebook-style):
```bash
--schedule sliding --episode_length 180 --episode_stride 30 --num_passes 3
```
Episodes are deterministic overlapping windows. Total episodes = `len(episode_starts) * num_passes`.

### DQN-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--double_dqn` | False | Enable Double DQN target computation |
| `--epsilon_start` | `0.3` | Initial exploration rate |
| `--epsilon_min` | `0.01` | Minimum exploration rate |
| `--epsilon_decay` | `0.995` | Per-episode epsilon decay factor |
| `--loss` | `mse` | Loss function: `mse` or `huber` |

### Rainbow-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_atoms` | `51` | Number of atoms in C51 return distribution |
| `--v_min` | `-10.0` | Minimum value of distributional support |
| `--v_max` | `10.0` | Maximum value of distributional support |
| `--n_step` | `3` | Multi-step return length |
| `--per_alpha` | `0.5` | Prioritized replay exponent (0=uniform, 1=full priority) |
| `--per_beta_start` | `0.4` | Initial importance sampling weight exponent |
| `--per_beta_end` | `1.0` | Final importance sampling weight exponent |
| `--per_beta_steps` | `100000` | Steps to linearly anneal beta |
| `--noisy_sigma` | `0.5` | Initial noise parameter for NoisyLinear layers |
| `--hidden_rainbow` | `128` | Hidden layer width for dueling streams |

### Training Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | `5e-4` | Learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--batch_size` | `64` | Mini-batch size |
| `--buffer_size` | `50000` | Replay buffer capacity |
| `--train_start` | `500` | Steps before training begins |
| `--target_update_freq` | `500` | Steps between target network updates |
| `--update_every` | `4` | Environment steps between each learning update |

## Important Notes

### Reward Schemes and Value Support

For **`portfolio_delta` reward** (raw dollar changes):
- Per-step rewards can be hundreds of dollars
- Use wider distributional support: `--v_min -500 --v_max 500` for Rainbow
- This prevents atom clipping and distribution collapse

For **`sharpe`/`sortino` rewards** (normalized):
- Default support `--v_min -10 --v_max 10` is sufficient

### Exploration Differences

**DQN/Double DQN**: Uses ε-greedy exploration
- `epsilon` starts high and decays each episode
- Training logs show current epsilon value

**Rainbow**: Uses noisy networks for exploration  
- No ε-greedy (epsilon always shows 0.0 in logs)
- Exploration via learned noise in NoisyLinear layers
- `select_action(..., explore=True)` enables noise, `explore=False` is deterministic

## Example Training Sessions

### Short Test Run
```bash
# Quick 50-episode test
python -m agents.value_based.train \
  --agent dqn --double_dqn \
  --ticker AAPL --features indicators --reward portfolio_delta \
  --episodes 50 --max_episode_steps 100 \
  --log_interval 5 --plot_every 1000000
```

### Notebook-Style Rainbow Training
```bash
# Replicate DQN notebook approach with Rainbow
python -m agents.value_based.train \
  --agent rainbow \
  --ticker AAPL --features indicators --reward portfolio_delta \
  --window_size 30 \
  --schedule sliding --episode_length 180 --episode_stride 30 --num_passes 3 \
  --lr 1e-4 --update_every 4 \
  --n_atoms 51 --v_min -500 --v_max 500 --n_step 3 \
  --train_start 1000 --buffer_size 50000 \
  --seed 0
```

### Multi-Ticker Comparison
```bash
# Train on different stocks
for ticker in AAPL GOOGL MSFT; do
  python -m agents.value_based.train \
    --agent rainbow --ticker $ticker \
    --features indicators --reward portfolio_delta \
    --episodes 300 --window_size 30
done
```

## Output and Checkpoints

**Training Output**:
```
Ticker: AAPL | Training data: 2516 days
State dim: 157 | Action dim: 3
Variant: rainbow | Schedule: random

Ep   10 | PV: $10,234.56 | Trades: 12 | Eps: 0.000 | Loss: 2.3456 | Buffer: 400
Ep   20 | PV: $9,876.54 | Trades: 8 | Eps: 0.000 | Loss: 1.9876 | Buffer: 800
...
Training complete! Checkpoint saved to runs/rainbow_AAPL_indicators_portfolio_delta
```

**Checkpoint Files**:
- DQN: `runs/{variant}_{ticker}_{features}_{reward}/dqn.pt`
- Rainbow: `runs/{variant}_{ticker}_{features}_{reward}/rainbow.pt`

Each checkpoint contains model weights, optimizer state, and training metadata.

## Next Steps

After training, evaluate your agents using the evaluation module:
```bash
# See evaluation/value_based/README.md for evaluation instructions
python -m evaluation.value_based.evaluate \
  --agent dqn --double_dqn \
  --checkpoint runs/double_dqn_AAPL_indicators_portfolio_delta \
  --ticker AAPL --features indicators --reward portfolio_delta \
  --window_size 30 --split test
```