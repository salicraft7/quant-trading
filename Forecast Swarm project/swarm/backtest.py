"""Walk-forward backtest with honest, online pooling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .aggregate import SkillWeighter, calibration_table, pool, score_table
from .swarm import ForecastSwarm, build_snapshot


def walk_forward(
    prices: pd.DataFrame,
    feats: pd.DataFrame,
    swarm: ForecastSwarm,
    horizon: int,
    step: int,
    min_train: int = 500,
    max_dates: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    close = prices["close"]
    # Only rows whose outcome is known can be scored; step apart by `step` rows.
    last = len(feats) - horizon - 1
    idx = list(range(min_train, last + 1, step))
    if max_dates:
        idx = idx[-max_dates:]

    rows = []
    for k, i in enumerate(idx):
        snap, _ = build_snapshot(feats, close, i, horizon)
        res = swarm.run(snap)
        row = {
            "date": feats.index[i],
            "row": i,
            "outcome": feats["label"].iloc[i],
            "fwd_ret": feats["fwd_ret"].iloc[i],
            "climatology": snap.base_rate,
            "stat_model": snap.model_prob,
        }
        row.update({f"agent_{n}": p for n, p in res.final.items()})
        row.update({f"r1_{n}": p for n, p in res.round1.items()})
        rows.append(row)
        if progress:
            print(f"[{k + 1}/{len(idx)}] {row['date'].date()}  model={snap.model_prob:.2f}  "
                  + " ".join(f"{n}={p:.2f}" for n, p in res.final.items()), flush=True)

    df = pd.DataFrame(rows)
    return add_pools(df, horizon)


def add_pools(df: pd.DataFrame, horizon: int, min_history: int = 30) -> pd.DataFrame:
    agents = [c for c in df.columns if c.startswith("agent_")]
    members = agents + ["stat_model"]
    df = df.copy()
    df["pool_equal"] = [pool(dict(zip(agents, r))) for r in df[agents].to_numpy()]
    df["pool_round1_equal"] = [pool(dict(zip(agents, r))) for r in df[[c.replace("agent_", "r1_") for c in agents]].to_numpy()]

    skill, alphas = [], []
    for k in range(len(df)):
        # Only forecasts already resolved at this forecast's time may inform the weights.
        resolved = df.iloc[:k][df["row"].iloc[:k] + horizon <= df["row"].iloc[k]]
        w = SkillWeighter(min_history=min_history).fit(resolved, members)
        skill.append(w(dict(zip(members, df[members].iloc[k]))))
        alphas.append(w.alpha)
    df["pool_skill_weighted"] = skill
    df["pool_alpha"] = alphas
    return df


def report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["climatology", "stat_model", "pool_round1_equal", "pool_equal", "pool_skill_weighted"]
    cols += [c for c in df.columns if c.startswith("agent_")]
    scores = score_table(df, cols)
    calib = calibration_table(df["pool_skill_weighted"], df["outcome"])
    return scores, calib


def bootstrap_brier_diff(df: pd.DataFrame, a: str, b: str, n: int = 2000, seed: int = 0) -> dict:
    """Is forecaster a really better than b, or is it noise? Bootstraps the Brier gap.

    Assumes step >= horizon, so forecast windows do not overlap and outcomes are roughly independent.
    """
    d = df.dropna(subset=["outcome"])
    y = d["outcome"].to_numpy(float)
    gap = (d[a].to_numpy() - y) ** 2 - (d[b].to_numpy() - y) ** 2
    rng = np.random.default_rng(seed)
    boots = np.array([gap[rng.integers(0, len(gap), len(gap))].mean() for _ in range(n)])
    return {"mean_gap": float(gap.mean()), "p_a_better": float((boots < 0).mean()),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
