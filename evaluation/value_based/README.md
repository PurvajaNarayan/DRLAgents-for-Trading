# Value-Based Agent Evaluation

This module provides evaluation tools for trained value-based agents (DQN, Double DQN, Rainbow) on validation and test sets.

## Quick Start

### Evaluate a Double DQN Agent

```bash
python -m evaluation.value_based.evaluate \
  --agent dqn --double_dqn \
  --checkpoint runs/double_dqn_AAPL_indicators_portfolio_delta \
  --ticker AAPL --features indicators --reward portfolio_delta \
  --window_size 30 --split test
```

### Evaluate a Rainbow Agent

```bash
python -m evaluation.value_based.evaluate \
  --agent rainbow \
  --checkpoint runs/rainbow_AAPL_indicators_portfolio_delta \
  --ticker AAPL --features indicators --reward portfolio_delta \
  --window_size 30 --split test \
  --n_atoms 51 --v_min -500 --v_max 500 --hidden_rainbow 128
```

## Command Line Arguments

### Core Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--agent` | No | `dqn` | Agent type: `dqn` or `rainbow` |
| `--checkpoint` | **Yes** | - | Path to checkpoint folder containing `dqn.pt` or `rainbow.pt` |
| `--ticker` | No | `AAPL` | Stock ticker symbol |
| `--features` | No | `indicators` | Feature type: `raw` or `indicators` |
| `--reward` | No | `portfolio_delta` | Reward scheme used during training |
| `--split` | No | `test` | Data split: `val` or `test` |
| `--window_size` | No | `30` | Lookback window (must match training) |

### DQN-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--double_dqn` | False | Set this flag if the DQN was trained with Double DQN |

### Rainbow-Specific Arguments

⚠️ **Critical**: These must exactly match the values used during training:

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_atoms` | `51` | Number of atoms in C51 distribution |
| `--v_min` | `-10.0` | Minimum value of distributional support |
| `--v_max` | `10.0` | Maximum value of distributional support |  
| `--hidden_rainbow` | `128` | Hidden layer width for dueling streams |

### Display Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--no_plot` | False | Skip plotting trading behavior (for headless runs) |

## Example Usage

### Basic Evaluation (with plot)

```bash
# DQN on test set
python -m evaluation.value_based.evaluate \
  --agent dqn \
  --checkpoint runs/dqn_AAPL_indicators_portfolio_delta \
  --ticker AAPL --features indicators --reward portfolio_delta \
  --window_size 30 --split test

# Rainbow on validation set  
python -m evaluation.value_based.evaluate \
  --agent rainbow \
  --checkpoint runs/rainbow_AAPL_indicators_portfolio_delta \
  --ticker AAPL --features indicators --reward portfolio_delta \
  --window_size 30 --split val \
  --n_atoms 51 --v_min -500 --v_max 500 --hidden_rainbow 128
```

### Headless Evaluation (no plot popup)

```bash
python -m evaluation.value_based.evaluate \
  --agent rainbow \
  --checkpoint runs/rainbow_AAPL_indicators_portfolio_delta \
  --ticker AAPL --features indicators --reward portfolio_delta \
  --window_size 30 --split test \
  --n_atoms 51 --v_min -500 --v_max 500 --hidden_rainbow 128 \
  --no_plot
```

### Batch Evaluation Script

```bash
#!/bin/bash
# Evaluate multiple checkpoints
CHECKPOINTS=(
  "runs/dqn_AAPL_indicators_portfolio_delta"
  "runs/double_dqn_AAPL_indicators_portfolio_delta" 
  "runs/rainbow_AAPL_indicators_portfolio_delta"
)

for checkpoint in "${CHECKPOINTS[@]}"; do
  echo "Evaluating $checkpoint..."
  
  if [[ $checkpoint == *"rainbow"* ]]; then
    python -m evaluation.value_based.evaluate \
      --agent rainbow --checkpoint "$checkpoint" \
      --ticker AAPL --features indicators --reward portfolio_delta \
      --window_size 30 --split test \
      --n_atoms 51 --v_min -500 --v_max 500 --hidden_rainbow 128 \
      --no_plot
  elif [[ $checkpoint == *"double_dqn"* ]]; then
    python -m evaluation.value_based.evaluate \
      --agent dqn --double_dqn --checkpoint "$checkpoint" \
      --ticker AAPL --features indicators --reward portfolio_delta \
      --window_size 30 --split test --no_plot
  else
    python -m evaluation.value_based.evaluate \
      --agent dqn --checkpoint "$checkpoint" \
      --ticker AAPL --features indicators --reward portfolio_delta \
      --window_size 30 --split test --no_plot
  fi
  echo "---"
done
```

## Output Format

### Console Output

```
TEST set: 415 days of AAPL
Loaded Rainbow DQN checkpoint from runs/rainbow_AAPL_indicators_portfolio_delta

============================================================
RAINBOW DQN PERFORMANCE (TEST SET)
============================================================
  Final Value:       $12,345.67
  Cumulative Return: +23.46%
  Sharpe Ratio:      1.2345
  Sortino Ratio:     1.5678  
  Max Drawdown:      8.90%
  Total Trades:      24

============================================================
BUY & HOLD BASELINE (TEST SET)
============================================================
  Final Value:       $11,234.56
  Cumulative Return: +12.35%

============================================================
Agent vs Buy & Hold: +11.11%
Agent OUTPERFORMS Buy & Hold
============================================================
```

### Performance Metrics

- **Final Value**: Ending portfolio value ($10,000 initial)
- **Cumulative Return**: Total return percentage  
- **Sharpe Ratio**: Risk-adjusted return (252-day annualized)
- **Sortino Ratio**: Downside risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Total Trades**: Number of sell transactions (completed trades)

### Visual Output (unless `--no_plot`)

Opens a matplotlib window showing:
- Stock price over the evaluation period
- Buy signals (green triangles)
- Sell signals (red triangles)  
- Final profit/loss in the title

## Critical Parameter Matching

⚠️ **The evaluation parameters MUST exactly match training parameters that affect observation shape:**

### Must Match Training

| Parameter | Why Critical | Error if Wrong |
|-----------|-------------|----------------|
| `--window_size` | Changes observation dimension | `RuntimeError: size mismatch for net.0.weight` |
| `--features` | Changes observation dimension | `RuntimeError: size mismatch for net.0.weight` |
| `--n_atoms` (Rainbow) | Changes network output shape | `RuntimeError: size mismatch` |
| `--v_min/v_max` (Rainbow) | Changes distributional support | Incorrect Q-value computation |
| `--hidden_rainbow` (Rainbow) | Changes network architecture | `RuntimeError: size mismatch` |

### Example Error

```
RuntimeError: Error(s) in loading state_dict for RainbowNetwork:
	size mismatch for net.0.weight: copying a param with shape torch.Size([128, 107]) 
	from checkpoint, the corresponding param in current model has shape torch.Size([128, 157]).
```

**Solution**: The checkpoint was trained with `window_size=20` (107-dim obs) but you're evaluating with `window_size=30` (157-dim obs). Use `--window_size 20`.

## Data Requirements

Evaluation expects CSV files in `data/processed/`:
- `{TICKER}_val.csv` - validation set
- `{TICKER}_test.csv` - test set

Files should have columns: `Open`, `High`, `Low`, `Close`, `Volume` with a datetime index.

## Troubleshooting

### Common Issues

1. **"No such file or directory: data/processed/AAPL_test.csv"**
   - Ensure your data files are in the correct location
   - Check the ticker name matches your file naming

2. **"RuntimeError: size mismatch for net.0.weight"**
   - Parameter mismatch between training and evaluation
   - Check `--window_size`, `--features`, and Rainbow architecture args

3. **"FileNotFoundError: [Errno 2] No such file: 'runs/.../dqn.pt'"** 
   - Checkpoint path is incorrect
   - For DQN, look for `dqn.pt`; for Rainbow, look for `rainbow.pt`

4. **Agent performs poorly (-99% return)**
   - The checkpoint might not have trained properly
   - Try a different checkpoint or retrain with more episodes

### Debug Mode

Add these Python lines to see detailed state information:
```python
# Add to evaluation script for debugging
print(f"Observation shape: {env.observation_space.shape}")
print(f"Action space: {env.action_space.n}")
print(f"Feature builder: {type(fb).__name__}")
```

## Comparison with Other Methods

The evaluation script automatically compares against a buy-and-hold baseline. For more comprehensive comparisons:

```bash
# Compare multiple evaluation scripts
python -m evaluation.compare_rewards  # Compare different reward schemes
python -m evaluation.compare_splits   # Compare val vs test performance
```

See the main `evaluation/` directory for additional comparison tools.