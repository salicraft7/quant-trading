"""Turning many probabilities into one, and scoring them.

The pool is a weighted average in log-odds space, optionally *extremized*
(multiplied by alpha > 1). Both the weights and alpha are learned only from
forecasts that had already resolved at the time of the new forecast, so the
backtest never peeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 0.01


def clip(p):
    return np.clip(p, EPS, 1 - EPS)


def logit(p):
    p = clip(np.asarray(p, float))
    return np.log(p / (1 - p))


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    p, y = clip(np.asarray(p, float)), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def pool(probs: dict[str, float], weights: dict[str, float] | None = None, alpha: float = 1.0) -> float:
    names = list(probs)
    w = np.array([1.0 if weights is None else weights.get(n, 0.0) for n in names])
    if w.sum() <= 0:
        w = np.ones(len(names))
    w = w / w.sum()
    z = alpha * float(w @ logit([probs[n] for n in names]))
    return float(1 / (1 + np.exp(-z)))


class SkillWeighter:
    """Weights members by past log loss and fits the extremizing factor alpha.

    Both are shrunk toward the safe default (equal weights, alpha = 1) because in noisy markets
    skill differences are small and estimates from a short history mostly fit noise. The shrinkage
    fades as resolved history grows (weight n / (n + prior_n)).
    """

    def __init__(self, min_history: int = 30, eta: float = 10.0, alphas=None, prior_n: int = 150):
        self.min_history = min_history
        self.eta = eta
        self.prior_n = prior_n
        self.alphas = alphas if alphas is not None else np.round(np.arange(0.5, 2.51, 0.1), 2)
        self.weights: dict[str, float] | None = None
        self.alpha = 1.0

    def fit(self, history: pd.DataFrame, members: list[str]) -> "SkillWeighter":
        h = history.dropna(subset=["outcome"])
        if len(h) < self.min_history:
            self.weights, self.alpha = None, 1.0
            return self
        y = h["outcome"].to_numpy(float)
        losses = np.array([log_loss(h[m], y) for m in members])
        trust = len(h) / (len(h) + self.prior_n)
        raw = np.exp(-self.eta * (losses - losses.min()))
        w = trust * raw / raw.sum() + (1 - trust) / len(members)
        self.weights = dict(zip(members, w))
        z = np.array([pool(dict(zip(members, row)), self.weights, 1.0) for row in h[members].to_numpy()])
        zl = logit(z)
        best = min(self.alphas, key=lambda a: log_loss(1 / (1 + np.exp(-a * zl)), y))
        self.alpha = float(1 + trust * (best - 1))
        return self

    def __call__(self, probs: dict[str, float]) -> float:
        return pool(probs, self.weights, self.alpha)


def calibration_table(p, y, bins: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = df.groupby("bin", observed=True).agg(n=("y", "size"), mean_forecast=("p", "mean"), hit_rate=("y", "mean"))
    return t.reset_index()


def score_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Brier, log loss, Brier skill vs. climatology, hit rate and a naive long/short P&L."""
    d = df.dropna(subset=["outcome"])
    y = d["outcome"].to_numpy(float)
    ref = brier(d["climatology"], y) if "climatology" in d else brier(np.full_like(y, y.mean()), y)
    rows = []
    for c in cols:
        p = d[c].to_numpy(float)
        pos = np.sign(p - 0.5)
        rows.append(
            {
                "forecaster": c,
                "brier": brier(p, y),
                "log_loss": log_loss(p, y),
                "brier_skill_vs_climatology": 1 - brier(p, y) / ref if ref > 0 else np.nan,
                "hit_rate": float(np.mean((p > 0.5) == (y > 0.5))),
                "mean_signed_fwd_ret": float(np.mean(pos * d["fwd_ret"].to_numpy(float))),
            }
        )
    return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)
