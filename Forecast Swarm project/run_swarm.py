"""Command line entry point.

Backtest on the NOK / Brent data in this repo with the offline mock (free, no API key):
    python run_swarm.py backtest --csv "../Oil Money project/data/brent crude nokjpy.csv" \
        --price-col nok --exog brent --horizon 5 --step 5 --llm mock

Same thing with Claude (needs ANTHROPIC_API_KEY; ~14 calls per date, cached on disk):
    python run_swarm.py backtest ... --llm claude --max-dates 60

Forecast the latest date:
    python run_swarm.py forecast --csv ... --price-col nok --exog brent --horizon 5 --llm claude
"""

from __future__ import annotations

import argparse
import json
import os

from swarm import (ClaudeLLM, ForecastSwarm, MockLLM, bootstrap_brier_diff, build_snapshot,
                   compute_features, load_price_csv, report, walk_forward)
from swarm.aggregate import pool


def make_llm(args):
    if args.llm == "mock":
        return MockLLM()
    return ClaudeLLM(model=args.model, effort=args.effort, cache_dir=os.path.join(args.out, "llm_cache"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["backtest", "forecast"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--price-col", required=True)
    ap.add_argument("--exog", nargs="*", default=[], help="related-market columns for the cross-asset analyst")
    ap.add_argument("--dayfirst", action="store_true")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--step", type=int, default=None, help="rows between forecasts (default = horizon)")
    ap.add_argument("--min-train", type=int, default=500)
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=2, choices=[1, 2])
    ap.add_argument("--llm", choices=["mock", "claude"], default="mock")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    prices = load_price_csv(args.csv, args.date_col, args.price_col, args.exog, args.dayfirst)
    feats = compute_features(prices, args.horizon)
    swarm = ForecastSwarm(make_llm(args), rounds=args.rounds)

    if args.mode == "forecast":
        i = len(feats) - 1
        snap, _ = build_snapshot(feats, prices["close"], i, args.horizon)
        res = swarm.run(snap)
        print(f"As of {feats.index[i].date()}, P(price higher in {args.horizon} days):")
        print(f"  climatology  {snap.base_rate:.3f}\n  stat model   {snap.model_prob:.3f}")
        for n, p in res.final.items():
            print(f"  {n:<12} {p:.3f}   (round 1: {res.round1[n]:.3f})")
        print(f"  EQUAL POOL   {pool(res.final):.3f}")
        print("  For skill-weighted pooling, run a backtest first: weights come from resolved forecasts.")
        with open(os.path.join(args.out, "latest_forecast.json"), "w") as fh:
            json.dump({"date": str(feats.index[i].date()), "snapshot": snap.__dict__, "rationales": res.rationales}, fh, indent=2, default=str)
        return

    df = walk_forward(prices, feats, swarm, args.horizon, args.step or args.horizon, args.min_train, args.max_dates)
    scores, calib = report(df)
    df.to_csv(os.path.join(args.out, "backtest_forecasts.csv"), index=False)
    scores.to_csv(os.path.join(args.out, "backtest_scores.csv"), index=False)

    print("\n=== Scores (lower Brier / log loss is better; climatology Brier is the bar to clear) ===")
    print(scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Calibration of the skill-weighted pool ===")
    print(calib.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for ref in ("climatology", "stat_model"):
        b = bootstrap_brier_diff(df, "pool_skill_weighted", ref)
        print(f"\nPool vs {ref}: mean Brier gap {b['mean_gap']:+.4f}, 95% CI {b['ci95'][0]:+.4f}..{b['ci95'][1]:+.4f}, "
              f"P(pool better) = {b['p_a_better']:.2f}")


if __name__ == "__main__":
    main()
