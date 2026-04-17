"""Visualization functions for trading agent evaluation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_behavior(
    prices: list[float] | np.ndarray,
    states_buy: list[int],
    states_sell: list[int],
    profit: float,
    episode: int | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot price chart with buy/sell markers.

    Args:
        prices:      List of closing prices.
        states_buy:  List of step indices where agent bought.
        states_sell: List of step indices where agent sold.
        profit:      Total profit for the episode.
        episode:     Episode number (optional, for title).
        save_path:   Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(prices, color="royalblue", lw=1.5, label="Close Price")

    # plot buy markers slightly below the price
    buy_prices = [prices[s] for s in states_buy if s < len(prices)]
    ax.scatter(
        states_buy,
        buy_prices,
        marker="^",
        color="lime",
        s=150,
        label=f"Buy ({len(states_buy)})",
        zorder=5,
        edgecolors="black",
        linewidths=0.8,
    )

    # plot sell markers slightly above the price
    sell_prices = [prices[s] for s in states_sell if s < len(prices)]
    ax.scatter(
        states_sell,
        sell_prices,
        marker="v",
        color="red",
        s=150,
        label=f"Sell ({len(states_sell)})",
        zorder=5,
        edgecolors="black",
        linewidths=0.8,
    )

    # draw lines connecting buy-sell pairs
    n_pairs = min(len(states_buy), len(states_sell))
    for i in range(n_pairs):
        b = states_buy[i]
        s = states_sell[i]
        if b < len(prices) and s < len(prices):
            color = "green" if prices[s] > prices[b] else "red"
            ax.plot([b, s], [prices[b], prices[s]], color=color, lw=1, ls="--", alpha=0.5)

    title = f"Total Profit: ${profit:,.2f}"
    if episode is not None:
        title = f"Episode {episode} | {title}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_reward_comparison(
    results: dict[str, list[float]],
    title: str = "Cumulative Reward — Reward Function Comparison",
    save_path: str | None = None,
) -> None:
    """
    Plot cumulative reward over trading days for different reward functions.

    Args:
        results: Dict mapping reward function name to list of per-step rewards.
    """
    colors = {"Simple": "#e74c3c", "Sharpe": "#3498db", "Sortino": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(15, 6))

    for name, rewards in results.items():
        cum_rewards = np.cumsum(rewards)
        color = colors.get(name)
        ax.plot(cum_rewards, lw=2, label=name, color=color)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Cumulative Reward")
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="black", lw=0.5, ls="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_agent_vs_baselines(
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    metric_name: str = "Cumulative Return",
    title: str = "Agent vs Baselines",
    save_path: str | None = None,
) -> None:
    """
    Bar chart comparing agent vs baselines on validation and test sets.
    """
    labels = list(val_metrics.keys())
    val_vals = [val_metrics[l] * 100 for l in labels]  # percentage
    test_vals = [test_metrics[l] * 100 for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, val_vals, width, label="Validation", color="#3498db")
    bars2 = ax.bar(x + width / 2, test_vals, width, label="Test", color="#e74c3c")

    ax.set_ylabel(f"{metric_name} (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", lw=0.5)

    def add_labels(bars):
        for bar in bars:
            h = bar.get_height()
            offset = 3 if h >= 0 else -12
            ax.annotate(f"{h:+.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, offset), textcoords="offset points", ha="center", fontsize=10)

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()

def _plot_single_set_bar(
    metrics: dict[str, float],
    baseline_value: float,
    set_name: str,
    ticker: str = "",
    save_path: str | None = None,
) -> None:
    labels = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]
    baseline_pct = baseline_value * 100

    # shades of blue for each agent
    blues = ["#1a3a5c", "#2e6b9e", "#5ba3d9", "#a0ccee"]
    bar_colors = [blues[i % len(blues)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.4)

    # Buy & Hold threshold line
    ax.axhline(
        y=baseline_pct, color="#e74c3c", lw=2, ls="--", zorder=4,
        label=f"Buy & Hold ({baseline_pct:+.1f}%)",
    )

    for bar, val in zip(bars, values):
        offset = 3 if val >= 0 else -15
        ax.annotate(
            f"{val:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center", fontsize=11, fontweight="bold", color="#1a3a5c",
        )

    title = f"{set_name} Set — Cumulative Return"
    if ticker:
        title = f"{ticker} {title}"
    ax.set_title(title, fontsize=13, fontweight="bold", color="#1a3a5c")
    ax.set_ylabel("Cumulative Return (%)", color="#1a3a5c")
    ax.tick_params(colors="#1a3a5c")
    ax.grid(alpha=0.15, axis="y")
    ax.axhline(y=0, color="#1a3a5c", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_val_vs_baseline(val_metrics, baseline_value, ticker="", save_path=None):
    """Bar chart: agents vs baseline threshold on VALIDATION set."""
    _plot_single_set_bar(val_metrics, baseline_value, "Validation", ticker, save_path)


def plot_test_vs_baseline(test_metrics, baseline_value, ticker="", save_path=None):
    """Bar chart: agents vs baseline threshold on TEST set."""
    _plot_single_set_bar(test_metrics, baseline_value, "Test", ticker, save_path)