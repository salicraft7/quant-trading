"""A plain statistical model that every LLM agent has to beat.

Ridge-penalised logistic regression fitted by Newton's method, in numpy only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticModel:
    def __init__(self, l2: float = 5.0, max_iter: int = 50):
        self.l2 = l2
        self.max_iter = max_iter
        self.cols: list[str] = []
        self.mu = self.sd = self.w = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask].to_numpy(float)
        self.cols = list(X.columns)
        self.mu = X.mean().to_numpy()
        self.sd = X.std().replace(0, 1).fillna(1).to_numpy()
        Z = np.c_[np.ones(len(X)), (X.to_numpy() - self.mu) / self.sd]
        reg = np.eye(Z.shape[1]) * self.l2
        reg[0, 0] = 0.0  # never shrink the intercept
        w = np.zeros(Z.shape[1])
        for _ in range(self.max_iter):
            p = sigmoid(Z @ w)
            g = Z.T @ (p - y) + reg @ w
            H = Z.T @ (Z * (p * (1 - p))[:, None]) + reg
            step = np.linalg.solve(H, g)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.w = w
        return self

    def predict_proba(self, x: pd.Series) -> float:
        v = x[self.cols].astype(float).to_numpy()
        v = np.where(np.isnan(v), self.mu, v)  # missing -> training mean (neutral)
        z = self.w[0] + ((v - self.mu) / self.sd) @ self.w[1:]
        return float(sigmoid(z))

    def top_drivers(self, x: pd.Series, k: int = 4) -> list[dict]:
        v = x[self.cols].astype(float).to_numpy()
        v = np.where(np.isnan(v), self.mu, v)
        contrib = ((v - self.mu) / self.sd) * self.w[1:]
        order = np.argsort(-np.abs(contrib))[:k]
        return [{"feature": self.cols[i], "log_odds_contribution": float(contrib[i])} for i in order]
