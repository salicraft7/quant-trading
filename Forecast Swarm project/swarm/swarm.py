"""Orchestration: build an anonymised snapshot, run the panel for two Delphi rounds."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .agents import Persona, Snapshot, default_personas, revision_prompt
from .baseline import LogisticModel
from .features import FEATURE_GROUPS, conditional_base_rates, cross_asset_columns, feature_columns


def build_snapshot(feats: pd.DataFrame, close: pd.Series, i: int, horizon: int, path_len: int = 30) -> tuple[Snapshot, LogisticModel]:
    """Snapshot for row i, using only labels that had resolved by row i (rows <= i - horizon)."""
    train = feats.iloc[: max(i - horizon + 1, 0)]
    train = train[train["label"].notna()]
    now = feats.iloc[i]
    cols = feature_columns(feats)
    usable = [c for c in cols if train[c].notna().mean() > 0.8]

    model = LogisticModel().fit(train[usable], train["label"])
    base_cols = FEATURE_GROUPS["trend"] + FEATURE_GROUPS["reversion"] + FEATURE_GROUPS["volatility"]
    path = close.iloc[max(0, i - path_len + 1) : i + 1].to_numpy(float)

    snap = Snapshot(
        horizon=horizon,
        features={c: (None if pd.isna(now[c]) else float(now[c])) for c in cols},
        price_path=list(path / path[-1] * 100),
        base_rate=float(train["label"].mean()),
        base_rate_n=len(train),
        conditional_rates=conditional_base_rates(train, now, [c for c in base_cols if c in usable]),
        model_prob=model.predict_proba(now),
        model_drivers=model.top_drivers(now),
        cross_asset_cols=[c for c in cross_asset_columns(feats) if c in usable],
    )
    return snap, model


@dataclass
class PanelResult:
    round1: dict[str, float]
    final: dict[str, float]
    rationales: dict[str, dict]


class ForecastSwarm:
    def __init__(self, llm, personas: list[Persona] | None = None, rounds: int = 2, max_workers: int = 8):
        self.llm = llm
        self.personas = personas
        self.rounds = rounds
        self.max_workers = max_workers

    def _personas(self, snap: Snapshot) -> list[Persona]:
        return self.personas or default_personas(bool(snap.cross_asset_cols))

    def _ctx(self, persona: Persona, snap: Snapshot, peer_probs=None) -> dict:
        return {
            "persona": persona.name,
            "features": snap.features,
            "base_rate": snap.base_rate,
            "conditional_rates": snap.conditional_rates,
            "model_prob": snap.model_prob,
            "peer_probs": peer_probs,
        }

    def run(self, snap: Snapshot) -> PanelResult:
        personas = self._personas(snap)

        def first(p: Persona):
            return self.llm.forecast(p.system_prompt(snap.horizon), p.evidence(snap), self._ctx(p, snap))

        with ThreadPoolExecutor(self.max_workers) as ex:
            r1 = dict(zip([p.name for p in personas], ex.map(first, personas)))

        final = r1
        if self.rounds >= 2:
            def second(p: Persona):
                peers = [o.model_dump() for n, o in r1.items() if n != p.name]
                user = p.evidence(snap) + "\n\n" + revision_prompt(r1[p.name].probability_up, r1[p.name].key_factors, peers)
                ctx = self._ctx(p, snap, peer_probs=[o["probability_up"] for o in peers])
                return self.llm.forecast(p.system_prompt(snap.horizon), user, ctx)

            with ThreadPoolExecutor(self.max_workers) as ex:
                final = dict(zip([p.name for p in personas], ex.map(second, personas)))

        clip = lambda x: float(np.clip(x, 0.01, 0.99))
        return PanelResult(
            round1={n: clip(o.probability_up) for n, o in r1.items()},
            final={n: clip(o.probability_up) for n, o in final.items()},
            rationales={n: o.model_dump() for n, o in final.items()},
        )
