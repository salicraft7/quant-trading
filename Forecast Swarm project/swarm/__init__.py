"""Forecast Swarm: a calibrated, backtested multi-agent forecaster."""

from .backtest import add_pools, bootstrap_brier_diff, report, walk_forward
from .features import compute_features, load_price_csv
from .llm import ClaudeLLM, MockLLM
from .swarm import ForecastSwarm, build_snapshot
