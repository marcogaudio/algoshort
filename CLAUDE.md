# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

`algoshort` is an algorithmic trading analysis system for Italian (FTSE MIB) and NASDAQ stocks. It generates per-ticker Jupyter notebooks, executes them via `papermill`, saves results as CSVs, and ranks trading opportunities.

## Commands

```bash
# Install package and dependencies
pip install .
pip install papermill jupyter nbconvert  # required for notebook execution

# Generate and execute notebooks for all tickers
python generate_notebooks.py --market italy       # 20 Italian tickers
python generate_notebooks.py --market nasdaq      # 25 NASDAQ tickers
python generate_notebooks.py --market italy --ticker ENI.MI   # single ticker
python generate_notebooks.py --market italy --max-tickers 5   # limit count

# Analyze results and rank top trading opportunities
python analyze_results.py --market italy --top 10

# Run tests
python -m unittest discover tests
```

## Architecture

The system has two layers:

**1. `algoshort/` package** — core trading logic:
- `yfinance_handler.py`: Downloads/caches OHLC data from Yahoo Finance (Parquet cache in `cache/`)
- `ohlcprocessor.py`: Computes relative OHLC prices (stock vs. benchmark)
- `signals.py`: Signal generation — `signal_rbo` (breakout), `signal_rtt` (turtle trader), `signal_rsma`/`signal_rema` (triple MA crossover)
- `regime_bo.py`: Breakout regime detection (`RegimeBO`)
- `regime_ma.py`: Triple Moving Average crossover regime (`TripleMACrossoverRegime`)
- `regime_fc.py`: Floor/Ceiling swing analysis (`RegimeFC`) using `scipy.signal.find_peaks`
- `combiner.py`: `HybridSignalCombiner` — combines entry/exit/direction signals; supports parallel grid search via `parallel_grid_search.py`
- `stop_loss.py`: ATR-based stop loss (`StopLossCalculator`)
- `position_sizing.py`: Position sizing based on risk tolerance
- `strategy_metrics.py`: Performance metrics (return, drawdown, risk-adjusted)
- `optimizer.py`: Parameter optimization
- `wrappers.py`: High-level `calculate_trading_edge()` entry point

**2. Root-level scripts:**
- `generate_notebooks.py`: Reads ticker lists, creates notebooks from an inline template, executes them with `papermill`, saves results to `results/{market}/`
- `analyze_results.py`: Loads all `*_equity.csv` and `*_signals.csv` from `results/`, scores strategies, prints top N trades
- `config.py`: **Single source of truth** for all strategy parameters (windows, thresholds, position sizing weights). Edit this to change strategy behavior globally.

## Data Flow

```
tickers_italy.txt / tickers_nasdaq.txt
  → generate_notebooks.py
    → YFinanceDataHandler (cache/ as Parquet)
    → OHLCProcessor (relative prices vs benchmark)
    → signals.py (rbo, rtt, rsma, rema, rrg)
    → HybridSignalCombiner (entry/exit/direction)
    → StopLossCalculator + PositionSizing
    → results/{market}/{TICKER}_equity.csv
               {TICKER}_signals.csv
  → analyze_results.py → ranked trade recommendations
```

## Markets & Config

- Italy: benchmark `FTSEMIB.MI`, tickers in `tickers_italy.txt`, outputs to `results/italy/` and `notebooks/italy/`
- NASDAQ: benchmark `^IXIC`, tickers in `tickers_nasdaq.txt`, outputs to `results/nasdaq/` and `notebooks/nasdaq/`
- Strategy parameters live in `config.py` — breakout windows, ATR multiplier, MA periods, position sizing, scoring weights, etc.

## CI/CD

GitHub Actions (`daily_analysis.yaml`) runs Mon–Fri at 21:00 UTC, processing both markets and committing results + notebooks to `main`. The `email_signals.yaml` workflow handles signal notifications.

## Signal Naming Convention

Signals use a prefix naming convention that encodes strategy type and parameters:
- `rbo_20` — relative breakout, 20-day window
- `rtt_5020` — relative turtle trader, 50-day entry / 20-day exit
- `rsma_50100150` — relative triple SMA crossover (50/100/150)
- `rema_50100150` — relative triple EMA crossover
- `rrg` — relative regime floor/ceiling
