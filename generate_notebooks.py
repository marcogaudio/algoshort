#!/usr/bin/env python3
"""
Generate and execute trading analysis notebooks for Italian stock market tickers.

This script:
1. Reads a list of Italian stock tickers
2. Creates a notebook for each ticker from a template
3. Executes the notebook with papermill
4. Saves the executed notebook with outputs

Usage:
    python generate_notebooks.py [--tickers-file tickers_italy.txt] [--output-dir notebooks/]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_tickers(tickers_file: str) -> List[str]:
    """
    Load ticker symbols from a text file.

    Args:
        tickers_file: Path to file with one ticker per line.

    Returns:
        List of ticker symbols.
    """
    tickers = []
    with open(tickers_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                tickers.append(line)

    logger.info(f"Loaded {len(tickers)} tickers from {tickers_file}")
    return tickers


def create_notebook_template() -> dict:
    """
    Create the notebook template structure.

    Returns:
        Notebook dictionary structure.
    """
    return {
        "cells": [
            # Cell 0: Markdown - Title and Overview
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
                    "Automated trading analysis for **{TICKER}** using the algoshort library.\n",
                    "\n",
                    "### Workflow:\n",
                    "1. Data Acquisition from Yahoo Finance\n",
                    "2. Relative Price Calculation vs FTSE MIB\n",
                    "3. Signal Generation (Breakout, Turtle, MA Crossover, Floor/Ceiling)\n",
                    "4. Returns and Equity Curve Calculation\n",
                    "5. Stop Loss and Position Sizing\n"
                ]
            },
            # Cell 1: Imports and Configuration
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# CONFIGURATION\n",
                    "# =============================================================================\n",
                    "TICKER = \"{TICKER}\"\n",
                    "BENCHMARK = \"FTSEMIB.MI\"\n",
                    "START_DATE = \"2016-01-01\"\n",
                    "INITIAL_CAPITAL = 100000\n",
                    "\n",
                    "# =============================================================================\n",
                    "# IMPORTS\n",
                    "# =============================================================================\n",
                    "import logging\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "from datetime import date\n",
                    "import warnings\n",
                    "\n",
                    "warnings.filterwarnings('ignore')\n",
                    "logging.basicConfig(level=logging.WARNING)\n",
                    "\n",
                    "print(f\"Analysis for: {TICKER}\")\n",
                    "print(f\"Benchmark: {BENCHMARK}\")\n",
                    "print(f\"Period: {START_DATE} to {date.today()}\")"
                ]
            },
            # Cell 2: Data Download
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 1. DATA ACQUISITION\n",
                    "# =============================================================================\n",
                    "from algoshort.yfinance_handler import YFinanceDataHandler\n",
                    "\n",
                    "handler = YFinanceDataHandler(cache_dir=\"./cache\", enable_logging=False)\n",
                    "\n",
                    "print(f\"Downloading {TICKER} and {BENCHMARK}...\")\n",
                    "handler.download_data(\n",
                    "    symbols=[TICKER, BENCHMARK],\n",
                    "    start=START_DATE,\n",
                    "    end=date.today().isoformat(),\n",
                    "    use_cache=True\n",
                    ")\n",
                    "\n",
                    "df = handler.get_ohlc_data(TICKER)\n",
                    "bmk = handler.get_ohlc_data(BENCHMARK)\n",
                    "df['fx'] = 1\n",
                    "\n",
                    "print(f\"Downloaded {len(df)} rows for {TICKER}\")\n",
                    "print(f\"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\")"
                ]
            },
            # Cell 3: Relative Prices
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 2. RELATIVE PRICE CALCULATION\n",
                    "# =============================================================================\n",
                    "from algoshort.ohlcprocessor import OHLCProcessor\n",
                    "\n",
                    "processor = OHLCProcessor()\n",
                    "df = processor.calculate_relative_prices(\n",
                    "    stock_data=df,\n",
                    "    benchmark_data=bmk,\n",
                    "    benchmark_column='close',\n",
                    "    rebase=True\n",
                    ")\n",
                    "\n",
                    "print(\"Relative OHLC columns created: ropen, rhigh, rlow, rclose\")"
                ]
            },
            # Cell 4: Signal Generation
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 3. SIGNAL GENERATION\n",
                    "# =============================================================================\n",
                    "from algoshort.regime_bo import RegimeBO\n",
                    "from algoshort.regime_ma import TripleMACrossoverRegime\n",
                    "from algoshort.regime_fc import RegimeFC\n",
                    "\n",
                    "# Breakout signals\n",
                    "regime_bo = RegimeBO(ohlc_stock=df)\n",
                    "for window in [20, 50, 100]:\n",
                    "    df = regime_bo.compute_regime(regime_type='breakout', window=window, relative=True, inplace=True)\n",
                    "\n",
                    "# Turtle Trader\n",
                    "df = regime_bo.compute_regime(regime_type='turtle', window=50, fast_window=20, relative=True, inplace=True)\n",
                    "\n",
                    "# Moving Average Crossover\n",
                    "regime_ma = TripleMACrossoverRegime(ohlc_stock=df)\n",
                    "for ma_type in ['sma', 'ema']:\n",
                    "    df = regime_ma.compute_ma_regime(\n",
                    "        ma_type=ma_type,\n",
                    "        short_window=50, medium_window=100, long_window=150,\n",
                    "        relative=True, inplace=True\n",
                    "    )\n",
                    "\n",
                    "# Floor/Ceiling Regime\n",
                    "regime_fc = RegimeFC(df=df, log_level=logging.WARNING)\n",
                    "df = regime_fc.compute_regime(relative=True, lvl=3, vlty_n=63, threshold=0.05, dgt=3, d_vol=1, dist_pct=0.05, retrace_pct=0.05, r_vol=1.0)\n",
                    "\n",
                    "# Collect signal columns\n",
                    "signal_columns = [col for col in df.columns\n",
                    "    if any(col.startswith(p) for p in ['rbo_', 'rtt_', 'rsma_', 'rema_', 'rrg'])\n",
                    "    and not any(k in col for k in ['short', 'medium', 'long', '_ch'])]\n",
                    "signal_columns = [x for x in signal_columns if x != \"rrg_ch\"]\n",
                    "\n",
                    "print(f\"Generated {len(signal_columns)} signals: {signal_columns}\")"
                ]
            },
            # Cell 5: Returns Calculation
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 4. RETURNS CALCULATION\n",
                    "# =============================================================================\n",
                    "from algoshort.returns import ReturnsCalculator\n",
                    "\n",
                    "returns_calc = ReturnsCalculator(\n",
                    "    ohlc_stock=df,\n",
                    "    open_col=\"open\", high_col=\"high\", low_col=\"low\", close_col=\"close\",\n",
                    "    relative_prefix=\"r\"\n",
                    ")\n",
                    "\n",
                    "df = returns_calc.get_returns_multiple(df=df, signals=signal_columns, relative=True, n_jobs=-1)\n",
                    "print(\"Returns calculated for all signals\")"
                ]
            },
            # Cell 6: Stop Loss
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 5. STOP LOSS CALCULATION\n",
                    "# =============================================================================\n",
                    "from algoshort.stop_loss import StopLossCalculator\n",
                    "\n",
                    "sl_calc = StopLossCalculator(df)\n",
                    "for signal in signal_columns:\n",
                    "    df = sl_calc.atr_stop_loss(signal=signal, window=14, multiplier=2.0)\n",
                    "    sl_calc.data = df\n",
                    "\n",
                    "print(f\"ATR stop losses calculated for {len(signal_columns)} signals\")"
                ]
            },
            # Cell 7: Position Sizing
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 6. POSITION SIZING & EQUITY CURVES\n",
                    "# =============================================================================\n",
                    "from algoshort.position_sizing import PositionSizing, run_position_sizing_parallel\n",
                    "\n",
                    "sizer = PositionSizing(\n",
                    "    tolerance=-0.10, mn=-0.0025, mx=-0.05,\n",
                    "    equal_weight=0.05, avg=0.03, lot=1,\n",
                    "    initial_capital=INITIAL_CAPITAL\n",
                    ")\n",
                    "\n",
                    "df = run_position_sizing_parallel(\n",
                    "    sizer=sizer, df=df, signals=signal_columns,\n",
                    "    chg_suffix=\"_chg1D_fx\", sl_suffix=\"_stop_loss\",\n",
                    "    close_col='close', n_jobs=-1, verbose=0\n",
                    ")\n",
                    "\n",
                    "print(\"Position sizing and equity curves calculated\")"
                ]
            },
            # Cell 8: Results Summary
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 7. RESULTS SUMMARY\n",
                    "# =============================================================================\n",
                    "print(\"=\" * 70)\n",
                    "print(f\"ANALYSIS SUMMARY - {TICKER}\")\n",
                    "print(\"=\" * 70)\n",
                    "\n",
                    "# Performance metrics\n",
                    "abs_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100\n",
                    "rel_return = (df['rclose'].iloc[-1] / df['rclose'].iloc[0] - 1) * 100\n",
                    "\n",
                    "print(f\"\\n1. DATA:\")\n",
                    "print(f\"   Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\")\n",
                    "print(f\"   Trading Days: {len(df):,}\")\n",
                    "\n",
                    "print(f\"\\n2. BUY & HOLD:\")\n",
                    "print(f\"   Absolute Return: {abs_return:+.2f}%\")\n",
                    "print(f\"   Relative Return (vs {BENCHMARK}): {rel_return:+.2f}%\")\n",
                    "\n",
                    "# Equity curve results\n",
                    "equity_cols = [col for col in df.columns if '_equity_equal' in col]\n",
                    "equity_results = []\n",
                    "for col in equity_cols:\n",
                    "    signal_name = col.replace('_equity_equal', '')\n",
                    "    final_equity = df[col].iloc[-1]\n",
                    "    total_return = (final_equity / INITIAL_CAPITAL - 1) * 100\n",
                    "    max_dd = (df[col].min() - df[col].max()) / df[col].max() * 100\n",
                    "    equity_results.append({\n",
                    "        'Signal': signal_name,\n",
                    "        'Final Equity': final_equity,\n",
                    "        'Return %': total_return,\n",
                    "        'Max DD %': max_dd\n",
                    "    })\n",
                    "\n",
                    "equity_df = pd.DataFrame(equity_results).sort_values('Return %', ascending=False)\n",
                    "\n",
                    "print(f\"\\n3. STRATEGY PERFORMANCE:\")\n",
                    "print(equity_df.to_string(index=False))\n",
                    "\n",
                    "# Best strategy\n",
                    "if len(equity_df) > 0:\n",
                    "    best = equity_df.iloc[0]\n",
                    "    print(f\"\\n4. BEST STRATEGY:\")\n",
                    "    print(f\"   Signal: {best['Signal']}\")\n",
                    "    print(f\"   Return: {best['Return %']:+.2f}%\")\n",
                    "    print(f\"   Final Equity: {best['Final Equity']:,.0f}\")\n",
                    "\n",
                    "print(f\"\\n5. DATAFRAME:\")\n",
                    "print(f\"   Shape: {df.shape}\")\n",
                    "print(f\"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\")\n",
                    "\n",
                    "print(\"\\n\" + \"=\" * 70)\n",
                    "print(\"ANALYSIS COMPLETE\")\n",
                    "print(\"=\" * 70)"
                ]
            },
            # Cell 9: Export Data
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# =============================================================================\n",
                    "# 8. EXPORT RESULTS\n",
                    "# =============================================================================\n",
                    "import os\n",
                    "\n",
                    "# Create output directory if needed\n",
                    "output_dir = \"./results\"\n",
                    "os.makedirs(output_dir, exist_ok=True)\n",
                    "\n",
                    "# Export equity summary\n",
                    "equity_df.to_csv(f\"{output_dir}/{TICKER.replace('.', '_')}_equity.csv\", index=False)\n",
                    "\n",
                    "# Export latest signals (last row)\n",
                    "latest = df.tail(1)[['date'] + signal_columns].copy()\n",
                    "latest['ticker'] = TICKER\n",
                    "latest.to_csv(f\"{output_dir}/{TICKER.replace('.', '_')}_signals.csv\", index=False)\n",
                    "\n",
                    "print(f\"Results exported to {output_dir}/\")\n",
                    "print(f\"  - {TICKER.replace('.', '_')}_equity.csv\")\n",
                    "print(f\"  - {TICKER.replace('.', '_')}_signals.csv\")"
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


def generate_notebook(ticker: str, template: dict, output_dir: str, execution_date: str) -> str:
    """
    Generate a notebook for a specific ticker.

    Args:
        ticker: Stock ticker symbol.
        template: Notebook template dictionary.
        output_dir: Directory to save the notebook.
        execution_date: Date string for the notebook.

    Returns:
        Path to the generated notebook.
    """
    import copy

    # Deep copy template to avoid mutations
    notebook = copy.deepcopy(template)

    # Replace placeholders in all cells
    for cell in notebook['cells']:
        if 'source' in cell:
            new_source = []
            for line in cell['source']:
                line = line.replace('{TICKER}', ticker)
                line = line.replace('{DATE}', execution_date)
                new_source.append(line)
            cell['source'] = new_source

    # Create output path
    output_path = os.path.join(output_dir, f"{ticker.replace('.', '_')}.ipynb")

    # Write notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    return output_path


def execute_notebook(notebook_path: str, output_path: Optional[str] = None, timeout: int = 600) -> bool:
    """
    Execute a notebook using papermill or nbconvert.

    Args:
        notebook_path: Path to the notebook to execute.
        output_path: Path for the executed notebook. If None, overwrites input.
        timeout: Execution timeout in seconds.

    Returns:
        True if execution succeeded, False otherwise.
    """
    if output_path is None:
        output_path = notebook_path

    try:
        # Try papermill first (preferred)
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
        # Fallback to nbconvert
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
        description='Generate and execute trading analysis notebooks for Italian stocks.'
    )
    parser.add_argument(
        '--tickers-file',
        default='tickers_italy.txt',
        help='Path to file with ticker symbols (default: tickers_italy.txt)'
    )
    parser.add_argument(
        '--output-dir',
        default='notebooks',
        help='Directory for generated notebooks (default: notebooks/)'
    )
    parser.add_argument(
        '--no-execute',
        action='store_true',
        help='Generate notebooks without executing them'
    )
    parser.add_argument(
        '--ticker',
        help='Process only a specific ticker (for testing)'
    )
    parser.add_argument(
        '--max-tickers',
        type=int,
        default=0,
        help='Maximum number of tickers to process (0 = all)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load tickers
    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = load_tickers(args.tickers_file)

    if args.max_tickers > 0:
        tickers = tickers[:args.max_tickers]

    # Create template
    template = create_notebook_template()
    execution_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Process each ticker
    success_count = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] Processing {ticker}...")

        try:
            # Generate notebook
            notebook_path = generate_notebook(ticker, template, args.output_dir, execution_date)
            logger.info(f"  Generated: {notebook_path}")

            # Execute if requested
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

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total tickers: {len(tickers)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_tickers)}")

    if failed_tickers:
        print(f"\nFailed tickers: {', '.join(failed_tickers)}")

    # Exit with error if any failed
    return 0 if not failed_tickers else 1


if __name__ == '__main__':
    sys.exit(main())
