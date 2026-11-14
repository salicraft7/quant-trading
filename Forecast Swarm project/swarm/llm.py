"""LLM back-ends: Claude (real) and a deterministic mock (offline tests, dry runs)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading

from pydantic import BaseModel


class AgentForecast(BaseModel):
    # Reasoning comes first in the schema so the model writes it before committing to a number.
    reasoning: str
    key_factors: list[str]
    probability_up: float


class ClaudeLLM:
    """Calls Claude with structured output, prompt caching and an on-disk response cache.

    The disk cache makes re-running a backtest free: identical prompts are never paid for twice.
    """

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "medium",
        cache_dir: str | None = "output/llm_cache",
        max_tokens: int = 16000,
    ):
        import anthropic  # imported lazily so the mock path needs no SDK

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, system: str, user: str) -> str | None:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(f"{self.model}|{self.effort}|{system}|{user}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def forecast(self, system: str, user: str, context: dict) -> AgentForecast:
        path = self._cache_path(system, user)
        if path and os.path.exists(path):
            with open(path) as fh:
                return AgentForecast(**json.load(fh))

        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            # The persona system prompt is identical across every date in a backtest: cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            output_format=AgentForecast,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the forecast request: {response.stop_details}")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise RuntimeError(f"No parsable forecast (stop_reason={response.stop_reason})")

        out = response.parsed_output
        if path:
            with self._lock, open(path, "w") as fh:
                json.dump(out.model_dump(), fh)
        return out


class MockLLM:
    """Rule-based stand-in for each persona. Lets the whole pipeline run without an API key.

    Its numbers are simple heuristics, not intelligence: use it to test plumbing, never to
    judge whether the LLM swarm adds value.
    """

    def forecast(self, system: str, user: str, context: dict) -> AgentForecast:
        f = context["features"]
        persona = context["persona"]
        base = context["base_rate"]

        def g(k, default=0.0):
            v = f.get(k)
            return default if v is None or (isinstance(v, float) and math.isnan(v)) else v

        def sq(z):
            return 1 / (1 + math.exp(-z))

        lo = math.log(base / (1 - base))
        if persona == "trend":
            p = sq(lo + 3.0 * g("ret_20") / max(g("vol_20", 0.2), 0.05) + 20 * g("macd_hist"))
        elif persona == "reversion":
            p = sq(lo - 0.25 * g("zscore_20") - 0.02 * (g("rsi_14", 50) - 50))
        elif persona == "volatility":
            p = sq(lo - 0.3 * max(g("vol_ratio", 1) - 1, 0) + 0.5 * g("drawdown"))
        elif persona == "base_rate":
            rates = [r["up_rate"] for r in context.get("conditional_rates", [])]
            p = sum(rates) / len(rates) if rates else base
        elif persona == "quant":
            p = context["model_prob"]
        elif persona == "cross_asset":
            xs = [v for k, v in f.items() if k.endswith("_ret_20") and v == v]
            p = sq(lo + (2.0 * sum(xs) / len(xs) if xs else 0.0))
        else:  # contrarian / anything else: lean against the crowd's obvious read
            p = sq(lo - 1.5 * g("ret_5"))

        peers = context.get("peer_probs")
        if peers:  # Delphi round: partial, not total, movement toward the peer median
            peers = sorted(peers)
            p = 0.75 * p + 0.25 * peers[len(peers) // 2]

        p = min(max(p, 0.02), 0.98)
        return AgentForecast(reasoning=f"mock {persona} heuristic", key_factors=[persona], probability_up=p)
