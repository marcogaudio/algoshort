#!/usr/bin/env python3
"""
Analyze trading results and suggest the top 5 best trades to perform.

This script:
1. Reads all equity and signal CSV files from results/
2. Analyzes strategy performance across all tickers
3. Identifies current active signals
4. Ranks and suggests the top 5 best trading opportunities

Usage:
    python analyze_results.py [--results-dir results/] [--top N]
"""

import argparse
import glob
import os
from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np


def load_all_results(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all equity, signals, and grid search results.

    Returns:
        Tuple of (equity_df, signals_df, grid_search_df)
    """
    # Load equity files
    equity_files = glob.glob(os.path.join(results_dir, "*_equity.csv"))
    equity_dfs = []
    for f in equity_files:
        ticker = os.path.basename(f).replace("_equity.csv", "").replace("_", ".")
        df = pd.read_csv(f)
        df["Ticker"] = ticker
        equity_dfs.append(df)

    equity_df = pd.concat(equity_dfs, ignore_index=True) if equity_dfs else pd.DataFrame()

    # Load signals files
    signals_files = glob.glob(os.path.join(results_dir, "*_signals.csv"))
    signals_dfs = []
    for f in signals_files:
        df = pd.read_csv(f)
        signals_dfs.append(df)

    signals_df = pd.concat(signals_dfs, ignore_index=True) if signals_dfs else pd.DataFrame()

    # Load grid search files
    grid_files = glob.glob(os.path.join(results_dir, "*_grid_search.csv"))
    grid_dfs = []
    for f in grid_files:
        ticker = os.path.basename(f).replace("_grid_search.csv", "").replace("_", ".")
        df = pd.read_csv(f)
        df["Ticker"] = ticker
        grid_dfs.append(df)

    grid_df = pd.concat(grid_dfs, ignore_index=True) if grid_dfs else pd.DataFrame()

    return equity_df, signals_df, grid_df


def analyze_best_strategies(equity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze and rank strategies by performance.

    Returns:
        DataFrame with best strategies ranked by Sharpe-like ratio
    """
    # Calculate performance metrics per ticker/signal
    equity_df = equity_df.copy()

    # Rename columns for consistency
    equity_df.columns = [c.strip() for c in equity_df.columns]

    # Calculate risk-adjusted return (Return / |Max Drawdown|)
    # Higher is better
    equity_df["Risk_Adjusted_Return"] = equity_df["Total Return"] / (abs(equity_df["Max Drawdown"]) + 0.01)

    # Rank by risk-adjusted return
    equity_df = equity_df.sort_values("Risk_Adjusted_Return", ascending=False)

    return equity_df


def get_current_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get current active signals (Long=1 or Short=-1).

    Returns:
        DataFrame with current signals per ticker
    """
    if signals_df.empty:
        return pd.DataFrame()

    # Get signal columns (exclude date and ticker)
    signal_cols = [c for c in signals_df.columns if c not in ["date", "ticker"]]

    # Melt to long format
    melted = signals_df.melt(
        id_vars=["ticker", "date"],
        value_vars=signal_cols,
        var_name="Signal",
        value_name="Position"
    )

    # Keep only active positions (Long or Short)
    active = melted[melted["Position"].isin([1, -1, 1.0, -1.0])].copy()
    active["Direction"] = active["Position"].apply(lambda x: "LONG" if x == 1 else "SHORT")

    return active


def calculate_composite_score(
    equity_df: pd.DataFrame,
    signals_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate a composite score for trading opportunities.

    Factors:
    - Strategy return (higher is better)
    - Max drawdown (lower absolute value is better)
    - Risk-adjusted return
    - Current signal strength (active signal is required)

    Returns:
        DataFrame with ranked trading opportunities
    """
    # Get active signals
    active_signals = get_current_signals(signals_df)

    if active_signals.empty or equity_df.empty:
        return pd.DataFrame()

    # Merge equity with active signals
    # equity_df has: Ticker, Signal, Final Equity, Total Return, Max Drawdown
    # active_signals has: ticker, Signal, Position, Direction

    merged = pd.merge(
        equity_df,
        active_signals,
        left_on=["Ticker", "Signal"],
        right_on=["ticker", "Signal"],
        how="inner"
    )

    if merged.empty:
        return pd.DataFrame()

    # Calculate composite score
    # Normalize each factor to 0-1 range

    # Return score (higher is better)
    return_min = merged["Total Return"].min()
    return_max = merged["Total Return"].max()
    if return_max > return_min:
        merged["Return_Score"] = (merged["Total Return"] - return_min) / (return_max - return_min)
    else:
        merged["Return_Score"] = 0.5

    # Drawdown score (lower absolute value is better)
    dd_min = merged["Max Drawdown"].min()  # Most negative
    dd_max = merged["Max Drawdown"].max()  # Least negative
    if dd_max > dd_min:
        merged["DD_Score"] = (merged["Max Drawdown"] - dd_min) / (dd_max - dd_min)
    else:
        merged["DD_Score"] = 0.5

    # Risk-adjusted score
    rar_min = merged["Risk_Adjusted_Return"].min()
    rar_max = merged["Risk_Adjusted_Return"].max()
    if rar_max > rar_min:
        merged["RAR_Score"] = (merged["Risk_Adjusted_Return"] - rar_min) / (rar_max - rar_min)
    else:
        merged["RAR_Score"] = 0.5

    # Composite score (weighted average)
    # 40% return, 30% drawdown, 30% risk-adjusted
    merged["Composite_Score"] = (
        0.40 * merged["Return_Score"] +
        0.30 * merged["DD_Score"] +
        0.30 * merged["RAR_Score"]
    )

    # Sort by composite score
    merged = merged.sort_values("Composite_Score", ascending=False)

    return merged


def get_top_trades(
    equity_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Get top N trading opportunities.

    Returns:
        DataFrame with top trades
    """
    # Analyze strategies
    equity_analyzed = analyze_best_strategies(equity_df)

    # Calculate composite scores
    ranked = calculate_composite_score(equity_analyzed, signals_df)

    if ranked.empty:
        return pd.DataFrame()

    # Select top N
    top_trades = ranked.head(top_n).copy()

    # Select relevant columns
    result_cols = [
        "Ticker", "Signal", "Direction", "Total Return", "Max Drawdown",
        "Risk_Adjusted_Return", "Composite_Score"
    ]

    result = top_trades[[c for c in result_cols if c in top_trades.columns]]

    return result


def get_date_range(signals_df: pd.DataFrame) -> Tuple[str, str]:
    """Get the date range from signals data."""
    if signals_df.empty or "date" not in signals_df.columns:
        return "N/A", "N/A"

    signals_df["date"] = pd.to_datetime(signals_df["date"])
    start_date = signals_df["date"].min().strftime("%Y-%m-%d")
    end_date = signals_df["date"].max().strftime("%Y-%m-%d")
    return start_date, end_date


def print_strategy_legend():
    """Print legend explaining each strategy."""
    print("\n" + "=" * 80)
    print("📖 STRATEGY LEGEND")
    print("=" * 80)

    strategies = {
        "rrg": "Relative Rotation Graph - Measures relative strength and momentum vs benchmark (FTSEMIB). Signals based on RRG quadrant position.",
        "rema_50100150": "Triple EMA Crossover - Uses 50/100/150 day Exponential Moving Averages. LONG when fast > medium > slow, SHORT when reversed.",
        "rtt_5020": "Turtle Trader Breakout (50/20) - LONG on 50-day high breakout, exit on 20-day low. SHORT on 50-day low breakdown.",
        "rtt_2010": "Turtle Trader Breakout (20/10) - Faster version using 20-day entry and 10-day exit signals.",
        "rbo_20": "Breakout 20-day - Simple breakout strategy based on 20-day high/low levels.",
        "rbo_55": "Breakout 55-day - Longer-term breakout strategy based on 55-day high/low levels.",
        "floor": "Floor Regime - Identifies price floor levels using volatility-adjusted support zones.",
        "ceiling": "Ceiling Regime - Identifies price ceiling levels using volatility-adjusted resistance zones.",
        "fc_regime": "Floor/Ceiling Combined - Combines floor and ceiling detection for regime identification.",
        "sma_cross": "SMA Crossover - Simple Moving Average crossover strategy.",
        "macd": "MACD Signal - Moving Average Convergence Divergence trend-following strategy.",
    }

    for code, description in strategies.items():
        print(f"\n  {code:15} {description}")

    print()


def print_assumptions(signals_df: pd.DataFrame):
    """Print analysis assumptions and parameters."""
    start_date, end_date = get_date_range(signals_df)

    print("\n" + "=" * 80)
    print("⚙️  ANALYSIS ASSUMPTIONS & PARAMETERS")
    print("=" * 80)
    print(f"\n  Data Period:")
    print(f"    Start Date:       {start_date}")
    print(f"    End Date:         {end_date}")
    print(f"\n  Benchmark:          FTSEMIB.MI (FTSE MIB Index)")
    print(f"\n  Position Sizing:")
    print(f"    Initial Capital:  €10,000")
    print(f"    Risk per Trade:   2%")
    print(f"    Stop Loss:        ATR-based (2x ATR)")
    print(f"\n  Scoring Weights:")
    print(f"    Total Return:     40%")
    print(f"    Max Drawdown:     30%")
    print(f"    Risk-Adj Return:  30%")
    print(f"\n  Signal Filter:      Only active signals (LONG=1 or SHORT=-1)")
    print()


def print_summary_report(
    equity_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    top_trades: pd.DataFrame
):
    """Print a formatted summary report."""

    print("=" * 80)
    print("ITALIAN STOCK MARKET - TRADING ANALYSIS SUMMARY")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Print assumptions first
    print_assumptions(signals_df)

    # Overall statistics
    n_tickers = equity_df["Ticker"].nunique()
    n_strategies = equity_df["Signal"].nunique()
    total_combinations = len(equity_df)

    print(f"\n📊 DATASET OVERVIEW")
    print("-" * 40)
    print(f"  Tickers analyzed:     {n_tickers}")
    print(f"  Strategies per ticker: {n_strategies}")
    print(f"  Total combinations:   {total_combinations}")

    # Current market sentiment
    active = get_current_signals(signals_df)
    if not active.empty:
        long_count = (active["Direction"] == "LONG").sum()
        short_count = (active["Direction"] == "SHORT").sum()
        total_active = len(active)

        print(f"\n📈 CURRENT MARKET SIGNALS")
        print("-" * 40)
        print(f"  Active signals: {total_active}")
        print(f"  LONG signals:   {long_count} ({100*long_count/total_active:.1f}%)")
        print(f"  SHORT signals:  {short_count} ({100*short_count/total_active:.1f}%)")

        if long_count > short_count:
            print(f"  Sentiment:      BULLISH")
        elif short_count > long_count:
            print(f"  Sentiment:      BEARISH")
        else:
            print(f"  Sentiment:      NEUTRAL")

    # Best performing strategies overall
    print(f"\n🏆 TOP PERFORMING STRATEGIES (by Return)")
    print("-" * 40)
    top_return = equity_df.nlargest(5, "Total Return")[["Ticker", "Signal", "Total Return", "Max Drawdown"]]
    for i, row in top_return.iterrows():
        print(f"  {row['Ticker']:12} | {row['Signal']:15} | Return: {row['Total Return']:+.2f}% | MaxDD: {row['Max Drawdown']:.2f}%")

    # Best risk-adjusted strategies
    print(f"\n🛡️  BEST RISK-ADJUSTED STRATEGIES")
    print("-" * 40)
    equity_df["RAR"] = equity_df["Total Return"] / (abs(equity_df["Max Drawdown"]) + 0.01)
    top_rar = equity_df.nlargest(5, "RAR")[["Ticker", "Signal", "Total Return", "Max Drawdown", "RAR"]]
    for i, row in top_rar.iterrows():
        print(f"  {row['Ticker']:12} | {row['Signal']:15} | Return: {row['Total Return']:+.2f}% | Sharpe-like: {row['RAR']:.2f}")

    # Top trades recommendation
    if not top_trades.empty:
        print(f"\n" + "=" * 80)
        print("🎯 TOP 5 RECOMMENDED TRADES")
        print("=" * 80)
        print("\nThese are the best trading opportunities based on:")
        print("  • Historical strategy performance (return)")
        print("  • Risk management (max drawdown)")
        print("  • Risk-adjusted return")
        print("  • Current active signal\n")

        for i, (_, row) in enumerate(top_trades.iterrows(), 1):
            direction_emoji = "🟢" if row["Direction"] == "LONG" else "🔴"
            print(f"  {i}. {direction_emoji} {row['Direction']:5} {row['Ticker']}")
            print(f"     Strategy:    {row['Signal']}")
            print(f"     Return:      {row['Total Return']:+.2f}%")
            print(f"     Max DD:      {row['Max Drawdown']:.2f}%")
            print(f"     Score:       {row['Composite_Score']:.3f}")
            print()

    # Summary table for copy/paste
    print("=" * 80)
    print("📋 SUMMARY TABLE (for easy copy/paste)")
    print("=" * 80)
    print(f"\n{'Rank':<6}{'Direction':<10}{'Ticker':<12}{'Strategy':<18}{'Return':<12}{'Max DD':<12}{'Score':<10}")
    print("-" * 80)
    for i, (_, row) in enumerate(top_trades.iterrows(), 1):
        print(f"{i:<6}{row['Direction']:<10}{row['Ticker']:<12}{row['Signal']:<18}{row['Total Return']:+.2f}%{'':<5}{row['Max Drawdown']:.2f}%{'':<5}{row['Composite_Score']:.3f}")

    # Print strategy legend
    print_strategy_legend()

    print("\n" + "=" * 80)
    print("⚠️  DISCLAIMER: Past performance does not guarantee future results.")
    print("    Always do your own research and manage risk appropriately.")
    print("=" * 80)


def export_results(top_trades: pd.DataFrame, output_dir: str):
    """Export results to CSV."""
    output_path = os.path.join(output_dir, "top_trades_recommendation.csv")
    top_trades.to_csv(output_path, index=False)
    print(f"\n💾 Results exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trading results and suggest top trades."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing result CSV files"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top trades to suggest (default: 5)"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export results to CSV"
    )

    args = parser.parse_args()

    # Load data
    print("Loading results...")
    equity_df, signals_df, grid_df = load_all_results(args.results_dir)

    if equity_df.empty:
        print(f"Error: No equity files found in {args.results_dir}")
        return 1

    print(f"Loaded {len(equity_df)} strategy results for {equity_df['Ticker'].nunique()} tickers")

    # Get top trades
    top_trades = get_top_trades(equity_df, signals_df, args.top)

    # Print report
    print_summary_report(equity_df, signals_df, top_trades)

    # Export if requested
    if args.export:
        export_results(top_trades, args.results_dir)

    return 0


if __name__ == "__main__":
    exit(main())
