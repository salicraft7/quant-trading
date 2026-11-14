"""Point-in-time feature engineering.

Every feature at row t uses only data up to and including t. The label at row t
(direction of the forward `horizon`-day return) is only known at t + horizon, and
the backtester respects that when it picks training rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Which features each kind of analyst gets to see. Giving agents *different*
# evidence (rather than one shared seed document) is what makes a crowd wiser
# than its members: errors only cancel out when they are not all the same error.
FEATURE_GROUPS: dict[str, list[str]] = {
    "trend": ["ret_5", "ret_20", "ret_60", "ma_gap_50", "ma_gap_200", "macd_hist"],
    "reversion": ["ret_1", "rsi_14", "zscore_20", "bb_pctb", "dist_52w_high"],
    "volatility": ["vol_20", "vol_ratio", "drawdown", "autocorr_20", "skew_60"],
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "ret_1": "1-day return",
    "ret_5": "5-day return",
    "ret_20": "20-day return",
    "ret_60": "60-day return",
    "ma_gap_50": "price / 50-day moving average - 1",
    "ma_gap_200": "price / 200-day moving average - 1",
    "macd_hist": "MACD histogram (12/26/9) divided by price",
    "rsi_14": "14-day RSI (0-100)",
    "zscore_20": "z-score of price vs its 20-day mean",
    "bb_pctb": "Bollinger %B (20, 2sd); 0 = lower band, 1 = upper band",
    "dist_52w_high": "price / 252-day high - 1",
    "vol_20": "annualised 20-day volatility",
    "vol_ratio": "20-day vol / 120-day vol",
    "drawdown": "drawdown from running peak",
    "autocorr_20": "lag-1 autocorrelation of daily returns over 20 days",
    "skew_60": "skewness of daily returns over 60 days",
}


def load_price_csv(
    path: str,
    date_col: str,
    price_col: str,
    exog_cols: list[str] | None = None,
    dayfirst: bool = False,
) -> pd.DataFrame:
    """Load a CSV into a date-indexed frame with a `close` column (+ exog columns)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst)
    df = df.set_index(date_col).sort_index()
    out = pd.DataFrame({"close": pd.to_numeric(df[price_col], errors="coerce")})
    for col in exog_cols or []:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out = out[~out.index.duplicated(keep="last")]
    return out.dropna(subset=["close"])


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def compute_features(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return features, the forward return and the binary label for every row."""
    close = prices["close"].astype(float)
    r = close.pct_change()
    f = pd.DataFrame(index=prices.index)

    for n in (1, 5, 20, 60):
        f[f"ret_{n}"] = close.pct_change(n)
    f["ma_gap_50"] = close / close.rolling(50).mean() - 1
    f["ma_gap_200"] = close / close.rolling(200, min_periods=120).mean() - 1
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    f["rsi_14"] = _rsi(close)
    m20, s20 = close.rolling(20).mean(), close.rolling(20).std()
    f["zscore_20"] = (close - m20) / s20
    f["bb_pctb"] = (close - (m20 - 2 * s20)) / (4 * s20)
    f["dist_52w_high"] = close / close.rolling(252, min_periods=60).max() - 1

    f["vol_20"] = r.rolling(20).std() * np.sqrt(252)
    f["vol_ratio"] = r.rolling(20).std() / r.rolling(120, min_periods=60).std()
    f["drawdown"] = close / close.cummax() - 1
    f["autocorr_20"] = r.rolling(20).apply(lambda x: pd.Series(x).autocorr(1), raw=True)
    f["skew_60"] = r.rolling(60).skew()

    # Cross-asset features for any extra columns (e.g. crude oil for a petro-currency).
    for col in prices.columns.drop("close"):
        x = prices[col].astype(float).ffill()
        xr = x.pct_change()
        f[f"{col}_ret_20"] = x.pct_change(20)
        f[f"{col}_corr_60"] = r.rolling(60).corr(xr)

    f["fwd_ret"] = close.shift(-horizon) / close - 1
    f["label"] = (f["fwd_ret"] > 0).astype(float).where(f["fwd_ret"].notna())
    return f.replace([np.inf, -np.inf], np.nan)


def feature_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in ("fwd_ret", "label")]


def cross_asset_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(f) if c.endswith(("_ret_20", "_corr_60"))]


def conditional_base_rates(
    train: pd.DataFrame, current: pd.Series, cols: list[str], n_buckets: int = 5
) -> list[dict]:
    """For each feature: in the past, when it sat in today's quintile, how often was the label 1?

    This is the "outside view" a superforecaster starts from, computed from data rather
    than recalled from memory.
    """
    out = []
    for col in cols:
        x = train[col]
        mask = x.notna() & train["label"].notna()
        if mask.sum() < n_buckets * 10 or pd.isna(current.get(col)):
            continue
        edges = np.unique(np.quantile(x[mask], np.linspace(0, 1, n_buckets + 1)))
        if len(edges) < 3:
            continue
        bucket = int(np.clip(np.searchsorted(edges, current[col], side="right") - 1, 0, len(edges) - 2))
        lo, hi = edges[bucket], edges[bucket + 1]
        in_bucket = mask & (x >= lo) & (x <= hi)
        n = int(in_bucket.sum())
        if n == 0:
            continue
        out.append(
            {
                "feature": col,
                "bucket": f"{bucket + 1} of {len(edges) - 1}",
                "range": [float(lo), float(hi)],
                "up_rate": float(train.loc[in_bucket, "label"].mean()),
                "n": n,
            }
        )
    return out
