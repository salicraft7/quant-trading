"""Analyst personas, the evidence each one sees, and the prompts they get."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .features import FEATURE_DESCRIPTIONS, FEATURE_GROUPS

COMMON_RULES = """You are one analyst in a panel that forecasts whether an anonymised asset's price
will be HIGHER in {horizon} trading days than it is today.

Rules that keep the panel honest:
- The asset name, the dates and the price level are hidden on purpose. Do not guess which
  asset or period this is, and do not use anything you remember about real market history.
  Reason only from the evidence you are given.
- Start from the base rate (how often this asset went up over {horizon} days in the past),
  then adjust for your evidence. Most evidence in liquid markets is weak: forecasts far from
  the base rate need strong, specific reasons.
- You are scored by the Brier score, so state your true probability. Do not round to 0.5 to be
  safe and do not exaggerate to sound decisive.
- Report probability_up as a number between 0 and 1."""

PERSONA_BRIEFS = {
    "trend": "You are a trend follower. You read momentum: multi-horizon returns, moving-average gaps and MACD.",
    "reversion": "You are a mean-reversion trader. You look for stretched prices: RSI, z-scores, Bollinger %B, distance from highs.",
    "volatility": "You are a risk and volatility analyst. You judge regime: volatility levels and shifts, drawdowns, autocorrelation, skew.",
    "base_rate": "You are a reference-class forecaster. You trust empirical frequencies: how often the asset rose in past situations like today's.",
    "quant": "You are a quantitative modeller. You read a fitted logistic regression and judge whether to trust it today.",
    "cross_asset": "You are a macro analyst. You read related markets and how strongly the asset has tracked them.",
    "contrarian": "You are the panel's contrarian. You look for the story everyone else will tell from this chart and ask how it fails.",
}


@dataclass
class Snapshot:
    """Everything known at the forecast time, already anonymised."""

    horizon: int
    features: dict
    price_path: list[float]  # last N closes rescaled so today's close = 100
    base_rate: float
    base_rate_n: int
    conditional_rates: list[dict] = field(default_factory=list)
    model_prob: float = 0.5
    model_drivers: list[dict] = field(default_factory=list)
    cross_asset_cols: list[str] = field(default_factory=list)


@dataclass
class Persona:
    name: str
    feature_cols: list[str]
    sees_price_path: bool = True
    sees_conditional_rates: bool = False
    sees_model: bool = False

    def system_prompt(self, horizon: int) -> str:
        return PERSONA_BRIEFS[self.name] + "\n\n" + COMMON_RULES.format(horizon=horizon)

    def evidence(self, s: Snapshot) -> str:
        parts = [f"Base rate: the asset rose over {s.horizon} days in {s.base_rate:.1%} of {s.base_rate_n} past cases."]
        cols = list(self.feature_cols)
        if self.name == "cross_asset":
            cols = cols + s.cross_asset_cols
        if cols:
            lines = []
            for c in cols:
                v = s.features.get(c)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                desc = FEATURE_DESCRIPTIONS.get(c) or _describe_cross(c)
                lines.append(f"- {c} = {v:.4g}  ({desc})")
            parts.append("Indicators today:\n" + "\n".join(lines))
        if self.sees_price_path:
            parts.append("Closing prices, oldest to newest, rescaled so today = 100:\n" + ", ".join(f"{p:.1f}" for p in s.price_path))
        if self.sees_conditional_rates and s.conditional_rates:
            rows = [
                f"- {r['feature']} in quintile {r['bucket']} (range {r['range'][0]:.3g} to {r['range'][1]:.3g}): "
                f"up {r['up_rate']:.1%} of {r['n']} times"
                for r in s.conditional_rates
            ]
            parts.append("Historical up-rate when each indicator was where it is today:\n" + "\n".join(rows))
        if self.sees_model:
            parts.append(
                f"Logistic regression (fitted on this asset's past only) says P(up) = {s.model_prob:.3f}.\n"
                f"Largest drivers (log-odds contributions): {json.dumps(s.model_drivers)}"
            )
        return "\n\n".join(parts)


def _describe_cross(col: str) -> str:
    if col.endswith("_ret_20"):
        return f"20-day return of related market '{col[:-7]}'"
    if col.endswith("_corr_60"):
        return f"60-day correlation of daily returns with related market '{col[:-8]}'"
    return col


def default_personas(has_cross_asset: bool) -> list[Persona]:
    t, r, v = FEATURE_GROUPS["trend"], FEATURE_GROUPS["reversion"], FEATURE_GROUPS["volatility"]
    personas = [
        Persona("trend", t),
        Persona("reversion", r),
        Persona("volatility", v),
        Persona("base_rate", [], sees_price_path=False, sees_conditional_rates=True),
        Persona("quant", [], sees_price_path=False, sees_model=True),
        Persona("contrarian", t[:3] + r[:3]),
    ]
    if has_cross_asset:
        personas.append(Persona("cross_asset", ["ret_20"]))
    return personas


def revision_prompt(own_prob: float, own_factors: list[str], peers: list[dict]) -> str:
    peer_lines = "\n".join(
        f"- Analyst {i + 1}: P(up) = {p['probability_up']:.2f}; factors: {'; '.join(p['key_factors'][:3])}"
        for i, p in enumerate(peers)
    )
    return (
        f"Round 2. Your first forecast was P(up) = {own_prob:.2f} (factors: {'; '.join(own_factors[:3])}).\n"
        f"The other analysts, who saw different evidence, said:\n{peer_lines}\n\n"
        "Revise only if they point to evidence you did not have and that bears on the question. "
        "Agreement alone is not evidence, and a panel that copies its median loses its value. "
        "Give your final probability."
    )
