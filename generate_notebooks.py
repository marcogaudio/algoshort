#!/usr/bin/env python3
"""
Generate and execute trading analysis notebooks for stock market tickers.

This script:
1. Reads a list of stock tickers (Italy or NASDAQ)
2. Creates a notebook for each ticker from the template
3. Executes the notebook with papermill
4. Saves the executed notebook with outputs

Usage:
    python generate_notebooks.py [--market italy|nasdaq] [--ticker AAPL]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import (
    MARKETS, DEFAULT_MARKET, START_DATE, INITIAL_CAPITAL,
    BREAKOUT_WINDOWS, TURTLE_ENTRY_WINDOW, TURTLE_EXIT_WINDOW,
    MA_SHORT, MA_MEDIUM, MA_LONG,
    FC_LEVEL, FC_VOLATILITY_WINDOW, FC_THRESHOLD, FC_RETRACEMENT, FC_DGT, FC_D_VOL, FC_DIST_PCT, FC_R_VOL,
    STOP_LOSS_ATR_WINDOW, STOP_LOSS_ATR_MULTIPLIER,
    POSITION_TOLERANCE, POSITION_MIN_RISK, POSITION_MAX_RISK, POSITION_EQUAL_WEIGHT, POSITION_AVG, POSITION_LOT
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_tickers(tickers_file: str) -> List[str]:
    """Load ticker symbols from a text file."""
    tickers = []
    with open(tickers_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                tickers.append(line)
    logger.info(f"Loaded {len(tickers)} tickers from {tickers_file}")
    return tickers


def create_notebook_template() -> dict:
    """Create the full notebook template matching ENI.MI.ipynb."""
    return {
        "cells": [
            # Cell 0: Title and Overview (Markdown)
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# {TICKER} - Algorithmic Trading Analysis\n",
                    "\n",
                    "**Generated:** {DATE}\n",
                    "\n",
                    "## Overview\n",
                    "\n",
                    "This notebook demonstrates a complete algorithmic trading workflow using the **algoshort** library.\n",
                    "\n",
                    "### Workflow Steps:\n",
                    "1. **Data Acquisition** - Download historical OHLC data from Yahoo Finance\n",
                    "2. **Relative Price Calculation** - Calculate performance relative to FTSE MIB benchmark\n",
                    "3. **Signal Generation** - Generate trading signals using multiple strategies\n",
                    "4. **Signal Combination** - Combine signals with entry/exit/direction logic\n",
                    "5. **Returns Calculation** - Calculate P&L and equity curves\n",
                    "6. **Stop Loss Calculation** - Implement risk management\n",
                    "7. **Position Sizing** - Determine optimal position sizes\n",
                    "\n",
                    "### Key Concepts:\n",
                    "- **Relative Prices**: Stock performance vs. market benchmark (isolates alpha)\n",
                    "- **Regime Detection**: Identify bullish/bearish market conditions\n",
                    "- **Signal Combination**: Use direction + entry + exit logic for robust signals\n",
                    "\n",
                    "---"
                ]
            },
            # Cell 1: Configuration Markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Configuration\n", "\n", "### Ticker Settings"]
            },
            # Cell 2: Configuration Code
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# TICKER CONFIGURATION\n",
                    "# =============================================================================\n",
                    "TICKER = \"{TICKER}\"\n",
                    "BENCHMARK = \"{BENCHMARK}\"\n",
                    "START_DATE = \"{START_DATE}\"\n",
                    "INITIAL_CAPITAL = {INITIAL_CAPITAL}"
                ]
            },
            # Cell 3: Imports Markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### Imports and Logging Setup"]
            },
            # Cell 4: Imports Code
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# IMPORTS\n",
                    "# =============================================================================\n",
                    "import logging\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from datetime import date\n",
                    "import warnings\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "logging.basicConfig(\n",
                    "    level=logging.WARNING,\n",
                    "    format=\"%(asctime)s [%(levelname)7s] %(name)s: %(message)s\",\n",
                    "    datefmt=\"%Y-%m-%d %H:%M:%S\",\n",
                    ")\n",
                    "\n",
                    "print(\"Imports completed successfully!\")"
                ]
            },
            # Cell 5: Data Acquisition Markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 1. Data Acquisition\n",
                    "\n",
                    "### YFinanceDataHandler\n",
                    "\n",
                    "The `YFinanceDataHandler` class provides:\n",
                    "- **Automatic caching** with parquet files (10-50x faster on repeated access)\n",
                    "- **Bulk downloads** with chunking to avoid rate limits\n",
                    "- **Data quality checks** and cleaning\n",
                    "\n",
                    "**Key Methods:**\n",
                    "- `download_data()` - Download data for symbols\n",
                    "- `get_ohlc_data()` - Get formatted OHLC data\n",
                    "- `get_info()` - Get company fundamentals"
                ]
            },
            # Cell 6: Data Download Code
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# DATA DOWNLOAD\n",
                    "# =============================================================================\n",
                    "from algoshort.yfinance_handler import YFinanceDataHandler\n",
                    "\n",
                    "handler = YFinanceDataHandler(cache_dir=\"./cache\", enable_logging=False, chunk_size=30)\n",
                    "\n",
                    "print(f\"Downloading data for {TICKER} and {BENCHMARK}...\")\n",
                    "print(f\"Period: {START_DATE} to {date.today()}\")\n",
                    "\n",
                    "handler.download_data(\n",
                    "    symbols=[TICKER, BENCHMARK],\n",
                    "    start=START_DATE,\n",
                    "    end=date.today().isoformat(),\n",
                    "    interval='1d',\n",
                    "    use_cache=True\n",
                    ")\n",
                    "\n",
                    "print(\"\\nDownload complete!\")"
                ]
            },
            # Cell 7: Data Quality Check
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# DATA QUALITY CHECK\n",
                    "# =============================================================================\n",
                    "print(\"Data Quality Summary:\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "summary = handler.list_available_data()\n",
                    "for symbol, info in summary.items():\n",
                    "    print(f\"\\n{symbol}:\")\n",
                    "    print(f\"  Rows: {info['rows']}\")\n",
                    "    print(f\"  Date Range: {info['date_range']}\")\n",
                    "    print(f\"  Missing Values: {info['missing_values']}\")"
                ]
            },
            # Cell 8: Get OHLC Data
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# GET OHLC DATA\n",
                    "# =============================================================================\n",
                    "df = handler.get_ohlc_data(TICKER)\n",
                    "bmk = handler.get_ohlc_data(BENCHMARK)\n",
                    "\n",
                    "# Add FX column (required for some calculations)\n",
                    "# Set to 1 for same currency\n",
                    "df['fx'] = 1\n",
                    "\n",
                    "print(f\"{TICKER} Data Shape: {df.shape}\")\n",
                    "print(f\"{BENCHMARK} Data Shape: {bmk.shape}\")\n",
                    "print(f\"\\nDate Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\")\n",
                    "print(f\"\\nFirst 5 rows:\")\n",
                    "df.head()"
                ]
            },
            # Cell 9: Company Info
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# COMPANY INFORMATION\n",
                    "# =============================================================================\n",
                    "info = handler.get_info(TICKER)\n",
                    "\n",
                    "print(f\"Company: {info.get('longName', 'N/A')}\")\n",
                    "print(f\"Sector: {info.get('sector', 'N/A')}\")\n",
                    "print(f\"Industry: {info.get('industry', 'N/A')}\")\n",
                    "print(f\"Market Cap: {info.get('marketCap', 0):,.0f}\")\n",
                    "print(f\"P/E Ratio: {info.get('trailingPE', 'N/A')}\")\n",
                    "print(f\"Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%\" if info.get('dividendYield') else \"Dividend Yield: N/A\")"
                ]
            },
            # Cell 10: Relative Prices Markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 2. Relative Price Calculation\n",
                    "\n",
                    "### OHLCProcessor\n",
                    "\n",
                    "**Why Relative Prices?**\n",
                    "\n",
                    "Relative prices show how a stock performs **compared to the market**:\n",
                    "- **Absolute Price**: Stock goes up 5% -> Could be market rally (beta)\n",
                    "- **Relative Price**: Stock goes up 5% vs market 2% -> Stock outperforms (alpha)\n",
                    "\n",
                    "**Calculation:**\n",
                    "```\n",
                    "Relative Price = Stock Price / Benchmark Price (rebased to 1.0)\n",
                    "```"
                ]
            },
            # Cell 11: Calculate Relative Prices
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# CALCULATE RELATIVE PRICES\n",
                    "# =============================================================================\n",
                    "from algoshort.ohlcprocessor import OHLCProcessor\n",
                    "\n",
                    "processor = OHLCProcessor()\n",
                    "df = processor.calculate_relative_prices(\n",
                    "    stock_data=df,\n",
                    "    benchmark_data=bmk,\n",
                    "    benchmark_column='close',\n",
                    "    digits=4,\n",
                    "    rebase=True\n",
                    ")\n",
                    "\n",
                    "print(\"Relative OHLC columns created:\")\n",
                    "print(\"  ropen  - Relative Open\")\n",
                    "print(\"  rhigh  - Relative High\")\n",
                    "print(\"  rlow   - Relative Low\")\n",
                    "print(\"  rclose - Relative Close\")\n",
                    "print(f\"\\nDataFrame shape: {df.shape}\")"
                ]
            },
            # Cell 12: Visualize Absolute vs Relative
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# VISUALIZE ABSOLUTE VS RELATIVE PERFORMANCE\n",
                    "# =============================================================================\n",
                    "fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)\n",
                    "\n",
                    "abs_normalized = (df['close'] / df['close'].iloc[0]) * 100\n",
                    "rel_normalized = (df['rclose'] / df['rclose'].iloc[0]) * 100\n",
                    "\n",
                    "axes[0].plot(df['date'], abs_normalized, 'b-', linewidth=1, label=f'{TICKER} (Absolute)')\n",
                    "axes[0].axhline(y=100, color='gray', linestyle='--', alpha=0.5)\n",
                    "axes[0].set_ylabel('Normalized Price (Base=100)')\n",
                    "axes[0].set_title(f'{TICKER} - Absolute Performance')\n",
                    "axes[0].legend()\n",
                    "axes[0].grid(True, alpha=0.3)\n",
                    "\n",
                    "axes[1].plot(df['date'], rel_normalized, 'g-', linewidth=1, label=f'{TICKER} vs {BENCHMARK} (Relative)')\n",
                    "axes[1].axhline(y=100, color='gray', linestyle='--', alpha=0.5)\n",
                    "axes[1].set_ylabel('Normalized Price (Base=100)')\n",
                    "axes[1].set_xlabel('Date')\n",
                    "axes[1].set_title(f'{TICKER} - Relative Performance vs {BENCHMARK}')\n",
                    "axes[1].legend()\n",
                    "axes[1].grid(True, alpha=0.3)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "abs_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100\n",
                    "rel_return = (df['rclose'].iloc[-1] / df['rclose'].iloc[0] - 1) * 100\n",
                    "print(f\"\\nPerformance Summary:\")\n",
                    "print(f\"  Absolute Return: {abs_return:+.2f}%\")\n",
                    "print(f\"  Relative Return (vs benchmark): {rel_return:+.2f}%\")"
                ]
            },
            # Cell 13: Signal Generation Markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 3. Signal Generation\n",
                    "\n",
                    "### Trading Signals Overview\n",
                    "\n",
                    "| Method | Class | Description |\n",
                    "|--------|-------|-------------|\n",
                    "| **Breakout** | `RegimeBO` | Price breaks above/below N-day high/low |\n",
                    "| **Turtle Trader** | `RegimeBO` | Dual-window breakout system |\n",
                    "| **MA Crossover** | `TripleMACrossoverRegime` | Triple moving average alignment |\n",
                    "| **Floor/Ceiling** | `RegimeFC` | Swing-based regime detection |\n",
                    "\n",
                    "**Signal Values:**\n",
                    "- `1` = Bullish / Long\n",
                    "- `0` = Neutral / Flat\n",
                    "- `-1` = Bearish / Short"
                ]
            },
            # Cell 14: Breakout Signals
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# BREAKOUT SIGNALS\n",
                    "# =============================================================================\n",
                    "from algoshort.regime_bo import RegimeBO\n",
                    "\n",
                    "regime_bo = RegimeBO(ohlc_stock=df)\n",
                    "breakout_windows = {BREAKOUT_WINDOWS}\n",
                    "\n",
                    "print(\"Generating Breakout Signals:\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "for window in breakout_windows:\n",
                    "    df = regime_bo.compute_regime(regime_type='breakout', window=window, relative=True, inplace=True)\n",
                    "    signal_col = f'rbo_{window}'\n",
                    "    counts = df[signal_col].value_counts().to_dict()\n",
                    "    print(f\"\\n  rbo_{window} ({window}-day breakout):\")\n",
                    "    print(f\"    Long (1):  {counts.get(1.0, 0):>5} bars ({counts.get(1.0, 0)/len(df)*100:.1f}%)\")\n",
                    "    print(f\"    Short (-1): {counts.get(-1.0, 0):>5} bars ({counts.get(-1.0, 0)/len(df)*100:.1f}%)\")"
                ]
            },
            # Cell 15: Turtle Trader Signals
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# TURTLE TRADER SIGNALS\n",
                    "# =============================================================================\n",
                    "print(\"\\nGenerating Turtle Trader Signals:\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "df = regime_bo.compute_regime(regime_type='turtle', window={TURTLE_ENTRY_WINDOW}, fast_window={TURTLE_EXIT_WINDOW}, relative=True, inplace=True)\n",
                    "\n",
                    "signal_col = 'rtt_{TURTLE_ENTRY_WINDOW}{TURTLE_EXIT_WINDOW}'\n",
                    "counts = df[signal_col].value_counts().to_dict()\n",
                    "print(f\"\\n  rtt_{TURTLE_ENTRY_WINDOW}{TURTLE_EXIT_WINDOW} (Slow={TURTLE_ENTRY_WINDOW}, Fast={TURTLE_EXIT_WINDOW}):\")\n",
                    "print(f\"    Long (1):   {counts.get(1, 0):>5} bars ({counts.get(1, 0)/len(df)*100:.1f}%)\")\n",
                    "print(f\"    Neutral (0): {counts.get(0, 0):>5} bars ({counts.get(0, 0)/len(df)*100:.1f}%)\")\n",
                    "print(f\"    Short (-1): {counts.get(-1, 0):>5} bars ({counts.get(-1, 0)/len(df)*100:.1f}%)\")"
                ]
            },
            # Cell 16: MA Crossover Signals
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# MOVING AVERAGE CROSSOVER SIGNALS\n",
                    "# =============================================================================\n",
                    "from algoshort.regime_ma import TripleMACrossoverRegime\n",
                    "\n",
                    "regime_ma = TripleMACrossoverRegime(ohlc_stock=df)\n",
                    "ma_params = {'short': {MA_SHORT}, 'medium': {MA_MEDIUM}, 'long': {MA_LONG}}\n",
                    "\n",
                    "print(\"\\nGenerating MA Crossover Signals:\")\n",
                    "print(\"=\" * 60)\n",
                    "print(f\"Parameters: Short={ma_params['short']}, Medium={ma_params['medium']}, Long={ma_params['long']}\")\n",
                    "\n",
                    "for ma_type in ['sma', 'ema']:\n",
                    "    df = regime_ma.compute_ma_regime(\n",
                    "        ma_type=ma_type,\n",
                    "        short_window=ma_params['short'],\n",
                    "        medium_window=ma_params['medium'],\n",
                    "        long_window=ma_params['long'],\n",
                    "        relative=True,\n",
                    "        inplace=True\n",
                    "    )\n",
                    "    signal_col = f\"r{ma_type}_{ma_params['short']}{ma_params['medium']}{ma_params['long']}\"\n",
                    "    counts = df[signal_col].value_counts().to_dict()\n",
                    "    print(f\"\\n  {signal_col} ({ma_type.upper()}):\")\n",
                    "    print(f\"    Long (1):  {counts.get(1.0, 0):>5} bars ({counts.get(1.0, 0)/len(df)*100:.1f}%)\")\n",
                    "    print(f\"    Short (-1): {counts.get(-1.0, 0):>5} bars ({counts.get(-1.0, 0)/len(df)*100:.1f}%)\")"
                ]
            },
            # Cell 17: Floor/Ceiling Regime
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# FLOOR/CEILING REGIME DETECTION\n",
                    "# =============================================================================\n",
                    "from algoshort.regime_fc import RegimeFC\n",
                    "\n",
                    "regime_fc = RegimeFC(df=df, log_level=logging.WARNING)\n",
                    "\n",
                    "print(\"\\nGenerating Floor/Ceiling Regime:\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "df = regime_fc.compute_regime(\n",
                    "    relative=True, lvl={FC_LEVEL}, vlty_n={FC_VOLATILITY_WINDOW}, threshold={FC_THRESHOLD},\n",
                    "    dgt={FC_DGT}, d_vol={FC_D_VOL}, dist_pct={FC_DIST_PCT}, retrace_pct={FC_RETRACEMENT}, r_vol={FC_R_VOL}\n",
                    ")\n",
                    "\n",
                    "if 'rrg' in df.columns:\n",
                    "    counts = df['rrg'].value_counts().to_dict()\n",
                    "    print(f\"\\n  rrg (Floor/Ceiling Regime):\")\n",
                    "    print(f\"    Bullish (1):  {counts.get(1.0, 0):>5} bars ({counts.get(1.0, 0)/len(df)*100:.1f}%)\")\n",
                    "    print(f\"    Bearish (-1): {counts.get(-1.0, 0):>5} bars ({counts.get(-1.0, 0)/len(df)*100:.1f}%)\")\n",
                    "\n",
                    "# Also compute absolute regime\n",
                    "df = regime_fc.compute_regime(\n",
                    "    relative=False, lvl={FC_LEVEL}, vlty_n={FC_VOLATILITY_WINDOW}, threshold={FC_THRESHOLD},\n",
                    "    dgt={FC_DGT}, d_vol={FC_D_VOL}, dist_pct={FC_DIST_PCT}, retrace_pct={FC_RETRACEMENT}, r_vol={FC_R_VOL}\n",
                    ")"
                ]
            },
            # Cell 18: Signal Summary
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# SIGNAL SUMMARY\n",
                    "# =============================================================================\n",
                    "signal_columns = [col for col in df.columns\n",
                    "    if any(col.startswith(prefix) for prefix in ['rbo_', 'rtt_', 'rsma_', 'rema_', 'rrg'])\n",
                    "    and not any(kw in col for kw in ['short', 'medium', 'long', '_ch'])]\n",
                    "signal_columns = [x for x in signal_columns if x != \"rrg_ch\"]\n",
                    "\n",
                    "print(f\"\\nTotal Signals Generated: {len(signal_columns)}\")\n",
                    "print(\"=\" * 60)\n",
                    "for sig in signal_columns:\n",
                    "    print(f\"  * {sig}\")"
                ]
            },
            # Cell 19: Signal Combination Markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 4. Signal Combination\n",
                    "\n",
                    "### Grid Search\n",
                    "\n",
                    "**Why Combine Signals?**\n",
                    "\n",
                    "Individual signals can be noisy. Combining them provides:\n",
                    "- **Direction Filter**: Only trade in direction of overall trend (floor/ceiling)\n",
                    "- **Entry Timing**: Use momentum signals for entry timing\n",
                    "- **Exit Timing**: Use different signal for exit"
                ]
            },
            # Cell 20: Grid Search
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# GRID SEARCH - TEST ALL COMBINATIONS\n",
                    "# =============================================================================\n",
                    "from algoshort.combiner import SignalGridSearch\n",
                    "\n",
                    "direction_col = 'rrg'\n",
                    "entry_exit_signals = [x for x in signal_columns if x != direction_col]\n",
                    "\n",
                    "print(\"Running Grid Search (All Entry/Exit Combinations):\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "searcher = SignalGridSearch(\n",
                    "    df=df.copy(),\n",
                    "    available_signals=entry_exit_signals,\n",
                    "    direction_col=direction_col\n",
                    ")\n",
                    "\n",
                    "results = searcher.run_grid_search_parallel(\n",
                    "    allow_flips=True,\n",
                    "    require_regime_alignment=True,\n",
                    "    n_jobs=-1,\n",
                    "    backend='multiprocessing'\n",
                    ")\n",
                    "\n",
                    "df = searcher.df"
                ]
            },
            # Cell 21: Update signal_columns with grid search outputs
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# UPDATE SIGNAL COLUMNS WITH COMBINED SIGNALS\n",
                    "# =============================================================================\n",
                    "# Add grid search combined signals to signal_columns\n",
                    "# so they are processed through returns, stop loss, and position sizing\n",
                    "signal_columns = results.output_column.tolist() + signal_columns"
                ]
            },
            # Cell 22: Grid Search Results
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# GRID SEARCH RESULTS\n",
                    "# =============================================================================\n",
                    "print(\"\\nGrid Search Results:\")\n",
                    "print(\"=\" * 60)\n",
                    "print(f\"Total Combinations Tested: {len(results)}\")\n",
                    "print(f\"Successful: {results['success'].sum()}\")\n",
                    "print(f\"Failed: {(~results['success']).sum()}\")\n",
                    "\n",
                    "print(\"\\nTop 10 Most Active Combinations:\")\n",
                    "top_results = results.nlargest(10, 'total_trades')[[\n",
                    "    'combination_name', 'total_trades', 'long_trades', 'short_trades', 'long_pct', 'short_pct'\n",
                    "]]\n",
                    "print(top_results.to_string(index=False))"
                ]
            },
            # Cell 23: Returns Markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 5. Returns Calculation\n",
                    "\n",
                    "### ReturnsCalculator\n",
                    "\n",
                    "**Calculates for each signal:**\n",
                    "- `{signal}_chg1D`: Daily price change x position\n",
                    "- `{signal}_PL_cum`: Cumulative P&L\n",
                    "- `{signal}_returns`: Percentage returns\n",
                    "- `{signal}_cumul`: Cumulative returns"
                ]
            },
            # Cell 24: Calculate Returns
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# CALCULATE RETURNS\n",
                    "# =============================================================================\n",
                    "from algoshort.returns import ReturnsCalculator\n",
                    "\n",
                    "returns_calc = ReturnsCalculator(\n",
                    "    ohlc_stock=df,\n",
                    "    open_col=\"open\", high_col=\"high\", low_col=\"low\", close_col=\"close\",\n",
                    "    relative_prefix=\"r\"\n",
                    ")\n",
                    "\n",
                    "print(\"Calculating Returns for All Signals:\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "df = returns_calc.get_returns_multiple(df=df, signals=signal_columns, relative=True, n_jobs=-1, verbose=False)\n",
                    "\n",
                    "print(f\"\\nReturn columns created for {len(signal_columns)} signals\")"
                ]
            },
            # Cell 25: Returns Summary
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# RETURNS SUMMARY\n",
                    "# =============================================================================\n",
                    "print(\"\\nCumulative Returns by Signal:\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "cumul_returns = {}\n",
                    "for sig in signal_columns:\n",
                    "    cumul_col = f'{sig}_cumul'\n",
                    "    if cumul_col in df.columns:\n",
                    "        cumul_returns[sig] = df[cumul_col].iloc[-1] * 100\n",
                    "\n",
                    "sorted_returns = sorted(cumul_returns.items(), key=lambda x: x[1], reverse=True)\n",
                    "\n",
                    "for sig, ret in sorted_returns:\n",
                    "    print(f\"  {sig}: {ret:+.2f}%\")"
                ]
            },
            # Cell 26: Stop Loss Markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 6. Stop Loss Calculation\n",
                    "\n",
                    "### StopLossCalculator\n",
                    "\n",
                    "**Available Methods:**\n",
                    "| Method | Description |\n",
                    "|--------|-------------|\n",
                    "| `atr_stop_loss` | ATR-based (volatility-adjusted) |\n",
                    "| `fixed_percentage_stop_loss` | Fixed % below/above entry |\n",
                    "| `breakout_channel_stop_loss` | N-day high/low |\n",
                    "| `moving_average_stop_loss` | MA-based stops |"
                ]
            },
            # Cell 27: Calculate Stop Losses
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# CALCULATE STOP LOSSES\n",
                    "# =============================================================================\n",
                    "from algoshort.stop_loss import StopLossCalculator\n",
                    "\n",
                    "sl_calc = StopLossCalculator(df)\n",
                    "\n",
                    "print(\"Calculating ATR-Based Stop Losses:\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "for signal in signal_columns:\n",
                    "    df = sl_calc.atr_stop_loss(signal=signal, window={STOP_LOSS_ATR_WINDOW}, multiplier={STOP_LOSS_ATR_MULTIPLIER})\n",
                    "    sl_calc.data = df\n",
                    "\n",
                    "sl_cols = [col for col in df.columns if col.endswith('_stop_loss')]\n",
                    "print(f\"\\nStop loss columns created: {len(sl_cols)}\")"
                ]
            },
            # Cell 28: Position Sizing Markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 7. Position Sizing\n",
                    "\n",
                    "### PositionSizing\n",
                    "\n",
                    "**Position Sizing Strategies:**\n",
                    "\n",
                    "| Strategy | Description | Risk Profile |\n",
                    "|----------|-------------|-------------|\n",
                    "| `equal_weight` | Fixed % of capital | Moderate |\n",
                    "| `constant` | Fixed risk per trade | Moderate |\n",
                    "| `concave` | Reduce risk as drawdown increases | Conservative |\n",
                    "| `convex` | Increase risk when winning | Aggressive |"
                ]
            },
            # Cell 29: Position Sizing
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# POSITION SIZING\n",
                    "# =============================================================================\n",
                    "from algoshort.position_sizing import PositionSizing, run_position_sizing_parallel\n",
                    "\n",
                    "sizer = PositionSizing(\n",
                    "    tolerance={POSITION_TOLERANCE}, mn={POSITION_MIN_RISK}, mx={POSITION_MAX_RISK},\n",
                    "    equal_weight={POSITION_EQUAL_WEIGHT}, avg={POSITION_AVG}, lot={POSITION_LOT},\n",
                    "    initial_capital=INITIAL_CAPITAL\n",
                    ")\n",
                    "\n",
                    "print(f\"Position Sizing Configuration:\")\n",
                    "print(\"=\" * 60)\n",
                    "print(f\"  Initial Capital: {INITIAL_CAPITAL:,}\")\n",
                    "print(f\"  Equal Weight: {int({POSITION_EQUAL_WEIGHT} * 100)}% per position\")\n",
                    "print(f\"  Risk Range: {int(abs({POSITION_MIN_RISK}) * 100)}% - {int(abs({POSITION_MAX_RISK}) * 100)}% per trade\")\n",
                    "print(f\"  Max Drawdown Tolerance: {int(abs({POSITION_TOLERANCE}) * 100)}%\")\n",
                    "\n",
                    "print(\"\\nCalculating Position Sizes...\")\n",
                    "\n",
                    "df = run_position_sizing_parallel(\n",
                    "    sizer=sizer, df=df, signals=signal_columns,\n",
                    "    chg_suffix=\"_chg1D_fx\", sl_suffix=\"_stop_loss\",\n",
                    "    close_col='close', n_jobs=-1, verbose=5\n",
                    ")"
                ]
            },
            # Cell 30: Equity Curve Analysis
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# EQUITY CURVE ANALYSIS\n",
                    "# =============================================================================\n",
                    "print(\"\\nEquity Curve Summary:\")\n",
                    "print(\"=\" * 60)\n",
                    "\n",
                    "# Find all equity columns (equal, constant, concave, convex)\n",
                    "equity_cols = [col for col in df.columns if '_equity_' in col]\n",
                    "\n",
                    "equity_results = []\n",
                    "for col in equity_cols:\n",
                    "    final_equity = df[col].iloc[-1]\n",
                    "    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100\n",
                    "    max_equity = df[col].max()\n",
                    "    max_drawdown = (df[col].min() - max_equity) / max_equity * 100\n",
                    "    equity_results.append({\n",
                    "        'equity_signal': col,\n",
                    "        'Final Equity': final_equity,\n",
                    "        'Total Return': total_return,\n",
                    "        'Max Drawdown': max_drawdown\n",
                    "    })\n",
                    "\n",
                    "equity_df = pd.DataFrame(equity_results).sort_values('Final Equity', ascending=False)\n",
                    "equity_df['combination_name'] = equity_df['equity_signal'].str.replace(r'_equity.*$', '', regex=True)\n",
                    "equity_df = equity_df.merge(results[['combination_name', 'total_trades']], on='combination_name', how='left').sort_values('Final Equity', ascending=False).reset_index(drop=True)\n",
                    "\n",
                    "# Create backward-compatible 'Signal' column for analyze_results.py\n",
                    "equity_df['Signal'] = equity_df['equity_signal']\n",
                    "\n",
                    "print(equity_df.head(20).to_string(index=False))"
                ]
            },
            # Cell 31: Visualize Equity Curves
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# VISUALIZE EQUITY CURVES\n",
                    "# =============================================================================\n",
                    "top_signals = equity_df.head(3)['equity_signal'].tolist()\n",
                    "\n",
                    "fig, ax = plt.subplots(figsize=(14, 6))\n",
                    "\n",
                    "for equity_col in top_signals:\n",
                    "    if equity_col in df.columns:\n",
                    "        ax.plot(df['date'], df[equity_col], label=equity_col, linewidth=1)\n",
                    "\n",
                    "ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')\n",
                    "\n",
                    "ax.set_xlabel('Date')\n",
                    "ax.set_ylabel('Equity')\n",
                    "ax.set_title(f'{TICKER} - Top 3 Strategy Equity Curves')\n",
                    "ax.legend(loc='best')\n",
                    "ax.grid(True, alpha=0.3)\n",
                    "ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            # Cell 32: Final Summary Markdown
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "\n",
                    "## 8. Final Summary\n",
                    "\n",
                    "### Complete Analysis Results"
                ]
            },
            # Cell 33: Final Summary
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# FINAL SUMMARY\n",
                    "# =============================================================================\n",
                    "print(\"=\" * 70)\n",
                    "print(f\"FINAL ANALYSIS SUMMARY - {TICKER}\")\n",
                    "print(\"=\" * 70)\n",
                    "\n",
                    "print(f\"\\n1. DATA:\")\n",
                    "print(f\"   Ticker: {TICKER}\")\n",
                    "print(f\"   Benchmark: {BENCHMARK}\")\n",
                    "print(f\"   Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\")\n",
                    "print(f\"   Trading Days: {len(df):,}\")\n",
                    "\n",
                    "abs_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100\n",
                    "rel_return = (df['rclose'].iloc[-1] / df['rclose'].iloc[0] - 1) * 100\n",
                    "print(f\"\\n2. BUY & HOLD PERFORMANCE:\")\n",
                    "print(f\"   Absolute Return: {abs_return:+.2f}%\")\n",
                    "print(f\"   Relative Return (vs {BENCHMARK}): {rel_return:+.2f}%\")\n",
                    "\n",
                    "best_strategy = equity_df.iloc[0]\n",
                    "print(f\"\\n3. BEST STRATEGY:\")\n",
                    "print(f\"   Signal: {best_strategy['equity_signal']}\")\n",
                    "print(f\"   Total Return: {best_strategy['Total Return']:+.2f}%\")\n",
                    "print(f\"   Final Equity: {best_strategy['Final Equity']:,.0f}\")\n",
                    "\n",
                    "print(f\"\\n4. DATAFRAME:\")\n",
                    "print(f\"   Shape: {df.shape}\")\n",
                    "print(f\"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\")\n",
                    "\n",
                    "ohlc_cols = len([c for c in df.columns if c in ['date', 'open', 'high', 'low', 'close', 'fx']])\n",
                    "rel_cols = len([c for c in df.columns if c.startswith('r') and c[1:] in ['open', 'high', 'low', 'close']])\n",
                    "signal_cols_count = len(signal_columns)\n",
                    "return_cols = len([c for c in df.columns if any(k in c for k in ['_chg1D', '_PL_cum', '_returns', '_cumul'])])\n",
                    "sl_cols_count = len([c for c in df.columns if '_stop_loss' in c])\n",
                    "equity_cols_count = len([c for c in df.columns if '_equity_' in c])\n",
                    "\n",
                    "print(f\"\\n5. COLUMNS BREAKDOWN:\")\n",
                    "print(f\"   OHLC: {ohlc_cols}\")\n",
                    "print(f\"   Relative OHLC: {rel_cols}\")\n",
                    "print(f\"   Signals: {signal_cols_count}\")\n",
                    "print(f\"   Returns: {return_cols}\")\n",
                    "print(f\"   Stop Losses: {sl_cols_count}\")\n",
                    "print(f\"   Equity Curves: {equity_cols_count}\")\n",
                    "\n",
                    "print(\"\\n\" + \"=\" * 70)\n",
                    "print(\"ANALYSIS COMPLETE!\")\n",
                    "print(\"=\" * 70)"
                ]
            },
            # Cell 34: Export Results
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# EXPORT RESULTS\n",
                    "# =============================================================================\n",
                    "import os\n",
                    "\n",
                    "output_dir = \"{RESULTS_DIR}\"\n",
                    "os.makedirs(output_dir, exist_ok=True)\n",
                    "\n",
                    "# Export equity summary\n",
                    "equity_df.to_csv(f\"{output_dir}/{TICKER.replace('.', '_')}_equity.csv\", index=False)\n",
                    "\n",
                    "# Export latest signals\n",
                    "latest = df.tail(1)[['date'] + signal_columns].copy()\n",
                    "latest['ticker'] = TICKER\n",
                    "latest.to_csv(f\"{output_dir}/{TICKER.replace('.', '_')}_signals.csv\", index=False)\n",
                    "\n",
                    "# Export grid search results\n",
                    "results.to_csv(f\"{output_dir}/{TICKER.replace('.', '_')}_grid_search.csv\", index=False)\n",
                    "\n",
                    "print(f\"Results exported to {output_dir}/\")\n",
                    "print(f\"  - {TICKER.replace('.', '_')}_equity.csv\")\n",
                    "print(f\"  - {TICKER.replace('.', '_')}_signals.csv\")\n",
                    "print(f\"  - {TICKER.replace('.', '_')}_grid_search.csv\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def generate_notebook(ticker: str, template: dict, output_dir: str, execution_date: str,
                      benchmark: str = None, results_dir: str = "./results") -> str:
    """Generate a notebook for a specific ticker."""
    import copy
    notebook = copy.deepcopy(template)

    # Use provided benchmark or default from config
    if benchmark is None:
        benchmark = MARKETS[DEFAULT_MARKET]["benchmark"]

    # Define all replacements from config
    replacements = {
        '{TICKER}': ticker,
        '{DATE}': execution_date,
        '{BENCHMARK}': benchmark,
        '{RESULTS_DIR}': results_dir,
        '{START_DATE}': START_DATE,
        '{INITIAL_CAPITAL}': str(INITIAL_CAPITAL),
        '{BREAKOUT_WINDOWS}': str(BREAKOUT_WINDOWS),
        '{TURTLE_ENTRY_WINDOW}': str(TURTLE_ENTRY_WINDOW),
        '{TURTLE_EXIT_WINDOW}': str(TURTLE_EXIT_WINDOW),
        '{MA_SHORT}': str(MA_SHORT),
        '{MA_MEDIUM}': str(MA_MEDIUM),
        '{MA_LONG}': str(MA_LONG),
        '{FC_LEVEL}': str(FC_LEVEL),
        '{FC_VOLATILITY_WINDOW}': str(FC_VOLATILITY_WINDOW),
        '{FC_THRESHOLD}': str(FC_THRESHOLD),
        '{FC_RETRACEMENT}': str(FC_RETRACEMENT),
        '{FC_DGT}': str(FC_DGT),
        '{FC_D_VOL}': str(FC_D_VOL),
        '{FC_DIST_PCT}': str(FC_DIST_PCT),
        '{FC_R_VOL}': str(FC_R_VOL),
        '{STOP_LOSS_ATR_WINDOW}': str(STOP_LOSS_ATR_WINDOW),
        '{STOP_LOSS_ATR_MULTIPLIER}': str(STOP_LOSS_ATR_MULTIPLIER),
        '{POSITION_TOLERANCE}': str(POSITION_TOLERANCE),
        '{POSITION_MIN_RISK}': str(POSITION_MIN_RISK),
        '{POSITION_MAX_RISK}': str(POSITION_MAX_RISK),
        '{POSITION_EQUAL_WEIGHT}': str(POSITION_EQUAL_WEIGHT),
        '{POSITION_AVG}': str(POSITION_AVG),
        '{POSITION_LOT}': str(POSITION_LOT),
    }

    for cell in notebook['cells']:
        if 'source' in cell:
            new_source = []
            for line in cell['source']:
                for placeholder, value in replacements.items():
                    line = line.replace(placeholder, value)
                new_source.append(line)
            cell['source'] = new_source

    output_path = os.path.join(output_dir, f"{ticker.replace('.', '_')}.ipynb")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    return output_path


def execute_notebook(notebook_path: str, output_path: Optional[str] = None, timeout: int = 600) -> bool:
    """Execute a notebook using papermill or nbconvert."""
    if output_path is None:
        output_path = notebook_path

    try:
        import papermill as pm
        pm.execute_notebook(
            notebook_path,
            output_path,
            kernel_name='python3',
            progress_bar=False,
            request_save_on_cell_execute=True
        )
        return True
    except ImportError:
        pass

    try:
        import subprocess
        result = subprocess.run([
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',
            f'--ExecutePreprocessor.timeout={timeout}',
            notebook_path
        ], capture_output=True, text=True, timeout=timeout + 60)

        if result.returncode != 0:
            logger.error(f"nbconvert error: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate and execute trading analysis notebooks for stocks.'
    )
    parser.add_argument('--market', default=DEFAULT_MARKET, choices=list(MARKETS.keys()),
                        help=f'Market to analyze (default: {DEFAULT_MARKET})')
    parser.add_argument('--tickers-file', default=None, help='Override tickers file')
    parser.add_argument('--output-dir', default=None, help='Override output directory')
    parser.add_argument('--no-execute', action='store_true')
    parser.add_argument('--ticker', help='Process only a specific ticker')
    parser.add_argument('--max-tickers', type=int, default=0)

    args = parser.parse_args()

    # Get market configuration
    market_config = MARKETS[args.market]
    benchmark = market_config["benchmark"]
    tickers_file = args.tickers_file or market_config["tickers_file"]
    output_dir = args.output_dir or market_config["output_dir"]
    results_dir = market_config["results_dir"]

    logger.info(f"Market: {market_config['name']}")
    logger.info(f"Benchmark: {benchmark}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = load_tickers(tickers_file)

    if args.max_tickers > 0:
        tickers = tickers[:args.max_tickers]

    template = create_notebook_template()
    execution_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    success_count = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] Processing {ticker}...")

        try:
            notebook_path = generate_notebook(ticker, template, output_dir, execution_date, benchmark, results_dir)
            logger.info(f"  Generated: {notebook_path}")

            if not args.no_execute:
                logger.info(f"  Executing...")
                if execute_notebook(notebook_path):
                    logger.info(f"  Success!")
                    success_count += 1
                else:
                    logger.error(f"  Execution failed!")
                    failed_tickers.append(ticker)
            else:
                success_count += 1

        except Exception as e:
            logger.error(f"  Error: {e}")
            failed_tickers.append(ticker)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total tickers: {len(tickers)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_tickers)}")

    if failed_tickers:
        print(f"\nFailed tickers: {', '.join(failed_tickers)}")

    return 0 if not failed_tickers else 1


if __name__ == '__main__':
    sys.exit(main())
