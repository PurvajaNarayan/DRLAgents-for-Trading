"""Download and prepare OHLCV data for the trading environment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"


def download_ohlcv(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2025-12-31",
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for a list of tickers and save to data/raw/.

    Returns a dict mapping ticker -> DataFrame.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dataframes = {}

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end, progress=False)

        # yfinance >=0.2.31 returns MultiIndex columns like ("Close", "AAPL")
        # flatten to just ["Open", "High", "Low", "Close", "Volume"]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index.name = "Date"
        df.dropna(inplace=True)

        path = RAW_DIR / f"{ticker}.csv"
        df.to_csv(path)
        dataframes[ticker] = df
        print(f"  Saved {len(df)} rows -> {path}")

    return dataframes


def prepare_splits(
    ticker: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV and split chronologically into train / val / test.

    Chronological split avoids look-ahead bias — critical for financial data.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{ticker}.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"No raw data for {ticker}. Run download_ohlcv first.")

    df = pd.read_csv(raw_path, index_col=0, parse_dates=["Date"])

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = PROCESSED_DIR / f"{ticker}_{split_name}.csv"
        split_df.to_csv(path)

    print(f"{ticker}: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ----- CLI usage -----
if __name__ == "__main__":
    # top 5 high-liquidity stocks as outlined in the proposal
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

    download_ohlcv(TICKERS)
    for t in TICKERS:
        prepare_splits(t)
