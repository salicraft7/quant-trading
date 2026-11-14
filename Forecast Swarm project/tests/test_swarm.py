import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swarm import ForecastSwarm, MockLLM, build_snapshot, compute_features, walk_forward  # noqa: E402
from swarm.aggregate import SkillWeighter, brier, pool  # noqa: E402
from swarm.baseline import LogisticModel  # noqa: E402


def _prices(n=900, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    oil = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    fx = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"close": fx, "oil": oil}, index=idx)


def test_features_do_not_look_ahead():
    p = _prices()
    full = compute_features(p, horizon=5)
    cut = compute_features(p.iloc[:600], horizon=5)
    cols = [c for c in full.columns if c not in ("fwd_ret", "label")]
    pd.testing.assert_frame_equal(full[cols].iloc[:600], cut[cols])


def test_snapshot_trains_only_on_resolved_labels():
    p = _prices()
    f = compute_features(p, horizon=10)
    # Poison every label that would not yet be known at row 700.
    f.loc[f.index[691:], "label"] = 1.0
    clean = compute_features(p, horizon=10)
    s1, _ = build_snapshot(f, p["close"], 700, 10)
    s2, _ = build_snapshot(clean, p["close"], 700, 10)
    assert s1.base_rate == s2.base_rate and s1.model_prob == s2.model_prob


def test_snapshot_is_anonymised():
    p = _prices()
    f = compute_features(p, horizon=5)
    s, _ = build_snapshot(f, p["close"], 700, 5)
    assert abs(s.price_path[-1] - 100) < 1e-9
    assert "2015" not in str(s.__dict__) and "2017" not in str(s.__dict__)


def test_logistic_recovers_signal():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=4000), "b": rng.normal(size=4000)})
    y = pd.Series((rng.random(4000) < 1 / (1 + np.exp(-2 * X["a"]))).astype(float))
    m = LogisticModel(l2=0.1).fit(X, y)
    assert m.w[1] > 1.5 and abs(m.w[2]) < 0.2


def test_pool_and_weighter():
    assert abs(pool({"a": 0.5, "b": 0.5}) - 0.5) < 1e-12
    assert pool({"a": 0.7, "b": 0.7}, alpha=2) > 0.7
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.5).astype(float)
    good = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.05, 400), 0.01, 0.99)
    noise = rng.random(400)
    h = pd.DataFrame({"outcome": y, "good": good, "noise": noise})
    w = SkillWeighter().fit(h, ["good", "noise"])
    assert w.weights["good"] > w.weights["noise"]


def test_end_to_end_mock_backtest():
    p = _prices()
    f = compute_features(p, horizon=5)
    df = walk_forward(p, f, ForecastSwarm(MockLLM()), horizon=5, step=5, min_train=500, progress=False)
    assert len(df) > 50
    for c in ("pool_equal", "pool_skill_weighted", "stat_model", "agent_cross_asset"):
        assert df[c].between(0, 1).all()
    assert brier(df["pool_equal"], df["outcome"]) < 0.3
