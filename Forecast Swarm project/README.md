# Forecast Swarm project

A panel of Claude analyst agents that forecasts **P(price is higher in N days)**, pools the
answers with weights earned on past performance, and **proves (or disproves) its accuracy in a
walk-forward backtest**.

The idea is in the same family as MiroFish (many LLM agents → one prediction), but the design aims
at measurable accuracy rather than a convincing simulated world.

| MiroFish-style swarm | This project |
|---|---|
| Thousands of personas simulate a social network; the outcome is read from the narrative | 6–7 analysts each output one **probability**, scored with the Brier score |
| Every agent reads the same seed documents → correlated errors | Each analyst sees a **different slice of evidence** (trend, mean reversion, volatility, base rates, a stat model, cross-asset), so errors partly cancel |
| No ground truth loop | **Walk-forward backtest**: features at t use data ≤ t, models train only on labels resolved by t |
| LLM may "remember" what happened | Asset name, dates and price level are **masked**; prices rescaled to 100, and agents are told not to use memory. Without this, an LLM backtest is contaminated |
| Opinion dynamics can herd | Two-round **Delphi**: agents see peers' anonymised forecasts once and are told agreement is not evidence |
| Equal voice for every agent | **Skill-weighted log-odds pool** with an extremizing factor, both learned only from already-resolved forecasts and shrunk toward equal weights when history is short |
| — | A plain **logistic regression** and **climatology** sit in the same table. The swarm only earns its API bill if it beats them |

## Run it

```bash
pip install -r requirements.txt

# Offline: the rule-based mock tests the plumbing, free, no API key
python run_swarm.py backtest --csv "../Oil Money project/data/brent crude nokjpy.csv" \
    --price-col nok --exog brent --horizon 5 --llm mock

# Real agents (needs ANTHROPIC_API_KEY). Start small: ~14 calls per date.
python run_swarm.py backtest --csv "../Oil Money project/data/brent crude nokjpy.csv" \
    --price-col nok --exog brent --horizon 5 --llm claude --max-dates 60

# Today's forecast
python run_swarm.py forecast --csv ... --price-col nok --exog brent --llm claude

python -m pytest tests
```

Outputs go to `output/`: every forecast (`backtest_forecasts.csv`), the score table, and a disk
cache of Claude responses so a rerun never pays twice.

## How to read the result

The score table ranks every forecaster by Brier score (0.25 = coin flip). Look at
`brier_skill_vs_climatology` (positive = better than always predicting the base rate), then at the
bootstrap line: "P(pool better) = 0.95" is evidence, 0.6 is noise. Short-horizon FX is close to a
random walk, so a real edge will be a skill score of a few percent, not a crystal ball. If the
LLM agents do not beat `stat_model`, the honest conclusion is that they add cost, not accuracy.

The mock numbers are heuristics, not intelligence; never use them to judge the swarm.

## Layout

```
swarm/features.py   point-in-time indicators, cross-asset features, conditional base rates
swarm/baseline.py   ridge logistic regression (numpy)
swarm/agents.py     personas, the evidence each sees, prompts
swarm/llm.py        Claude client (structured output, caching, refusal fallback) + mock
swarm/swarm.py      anonymised snapshot + two-round Delphi panel
swarm/aggregate.py  log-odds pooling, skill weighting, Brier / log loss / calibration
swarm/backtest.py   walk-forward loop, online pooling, bootstrap significance test
```

## Next steps worth trying

- Use the Message Batches API for backtests (half price, no latency needs).
- Add news or text evidence. It must be timestamped and filtered to before the forecast date.
- Try more horizons and assets before trusting any one result. One series is one sample.
