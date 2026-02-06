"""
=============================================================================
ALGOSHORT LIBRARY - COMPREHENSIVE USAGE EXAMPLE
=============================================================================

This script demonstrates ALL modules and methods of the algoshort library
for algorithmic trading analysis. It covers the complete workflow:

1. Data Acquisition (YFinanceDataHandler)
2. Data Processing (OHLCProcessor)
3. Signal Generation (RegimeBO, TripleMACrossoverRegime, RegimeFC)
4. Signal Combination (SignalGridSearch, HybridSignalCombiner)
5. Returns Calculation (ReturnsCalculator)
6. Stop Loss Calculation (StopLossCalculator)
7. Position Sizing (PositionSizing)
8. Strategy Metrics (StrategyMetrics)
9. Visualization (plots)

Author: AlgoShort Team
Version: 0.1.0
=============================================================================
"""

import logging
import pandas as pd
import numpy as np
from datetime import date

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Set up logging to control verbosity of library output
logging.basicConfig(
    level=logging.WARNING,  # Change to DEBUG for more detail, ERROR for less
    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def test_yfinance_handler():
    """
    ==========================================================================
    MODULE 1: YFinanceDataHandler
    ==========================================================================

    Purpose: Download, cache, and manage financial market data from Yahoo Finance.

    Key Features:
    - Automatic chunking for bulk downloads (avoids rate limits)
    - Intelligent caching with parquet files (10-50x faster repeated access)
    - Built-in data cleaning and quality checks
    - Multiple output formats (wide, long, OHLC)

    Main Methods:
    - download_data(): Download data for one or more symbols
    - get_data(): Retrieve data from memory
    - get_ohlc_data(): Get OHLC formatted data
    - get_combined_data(): Get multiple symbols in long format
    - get_multiple_symbols_data(): Get multiple symbols in wide format
    - get_info(): Get company metadata
    - list_available_data(): Summary of data in memory
    - save_data(): Export data to CSV/Excel/Parquet
    - clear_cache(): Remove cached files
    """
    print("\n" + "=" * 70)
    print("MODULE 1: YFinanceDataHandler - Data Acquisition")
    print("=" * 70)

    from algoshort.yfinance_handler import YFinanceDataHandler

    # ---------------------------------------------------------------------
    # Initialize the handler with caching enabled
    # ---------------------------------------------------------------------
    # cache_dir: Directory to store parquet cache files
    # enable_logging: Show download progress and cache hits/misses
    # chunk_size: Symbols per batch (lower = less rate limiting issues)
    handler = YFinanceDataHandler(
        cache_dir="./cache",
        enable_logging=False,  # Set True for verbose output
        chunk_size=30
    )

    # ---------------------------------------------------------------------
    # Define symbols to download
    # ---------------------------------------------------------------------
    # Using Italian stocks as example (you can change to US stocks like AAPL, MSFT)
    stock_symbol = "A2A.MI"
    benchmark_symbol = "FTSEMIB.MI"

    # ---------------------------------------------------------------------
    # Download historical data
    # ---------------------------------------------------------------------
    # Options:
    # - period: '1y', '2y', '5y', 'max' OR use start/end dates
    # - interval: '1d' (daily), '1wk' (weekly), '1h' (hourly)
    # - use_cache: True to use cached data if available
    print(f"\n1. Downloading data for {stock_symbol} and {benchmark_symbol}...")

    handler.download_data(
        symbols=[stock_symbol, benchmark_symbol],
        start='2016-01-01',          # Start date
        end=date.today().isoformat(), # End date (today)
        interval='1d',                # Daily data
        use_cache=True                # Use cache if available
    )

    # ---------------------------------------------------------------------
    # Check data quality
    # ---------------------------------------------------------------------
    print("\n2. Data Quality Summary:")
    summary = handler.list_available_data()
    for symbol, info in summary.items():
        print(f"   {symbol}: {info['rows']} rows, {info['date_range']}, "
              f"missing: {info['missing_values']}")

    # ---------------------------------------------------------------------
    # Get OHLC data (formatted for technical analysis)
    # ---------------------------------------------------------------------
    print("\n3. Getting OHLC data...")
    df = handler.get_ohlc_data(stock_symbol)
    print(f"   Stock data shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

    bmk = handler.get_ohlc_data(benchmark_symbol)
    print(f"   Benchmark data shape: {bmk.shape}")

    # ---------------------------------------------------------------------
    # Get company information (fundamental data)
    # ---------------------------------------------------------------------
    print("\n4. Company Information:")
    info = handler.get_info(stock_symbol)
    print(f"   Name: {info.get('longName', 'N/A')}")
    print(f"   Sector: {info.get('sector', 'N/A')}")
    print(f"   Industry: {info.get('industry', 'N/A')}")

    # Add FX column (required for some calculations, set to 1 for same currency)
    df['fx'] = 1

    return handler, df, bmk


def test_ohlc_processor(df, bmk):
    """
    ==========================================================================
    MODULE 2: OHLCProcessor
    ==========================================================================

    Purpose: Process OHLC data and calculate relative prices vs benchmark.

    Key Features:
    - Calculate relative strength of an asset against a benchmark
    - Automatic normalization of column names
    - Built-in data quality checks

    Main Methods:
    - calculate_relative_prices(): Divide OHLC by benchmark (rebased)

    Why Relative Prices?
    - Isolates stock-specific performance from market movements
    - Useful for pair trading and market-neutral strategies
    - Helps identify true outperformers/underperformers
    """
    print("\n" + "=" * 70)
    print("MODULE 2: OHLCProcessor - Relative Price Calculation")
    print("=" * 70)

    from algoshort.ohlcprocessor import OHLCProcessor

    # ---------------------------------------------------------------------
    # Initialize the processor
    # ---------------------------------------------------------------------
    processor = OHLCProcessor()

    # ---------------------------------------------------------------------
    # Calculate relative prices
    # ---------------------------------------------------------------------
    # Divides each OHLC value by the benchmark, rebasing to 1.0 at start
    # This shows how the stock performs RELATIVE to the market
    print("\n1. Calculating relative prices...")

    df = processor.calculate_relative_prices(
        stock_data=df,
        benchmark_data=bmk,
        benchmark_column='close',  # Use benchmark close prices
        digits=4,                   # Decimal places for rounding
        rebase=True                 # Rebase benchmark to 1.0 at start
    )

    # ---------------------------------------------------------------------
    # Show results
    # ---------------------------------------------------------------------
    print("\n2. New columns added (relative OHLC):")
    rel_cols = [col for col in df.columns if col.startswith('r') and col[1:] in ['open', 'high', 'low', 'close']]
    print(f"   {rel_cols}")

    print("\n3. Sample of absolute vs relative close prices:")
    print(df[['date', 'close', 'rclose']].head())

    print("\n4. Correlation between absolute and relative prices:")
    corr = df['close'].corr(df['rclose'])
    print(f"   Correlation: {corr:.4f}")
    print("   (Lower correlation = more diversification benefit)")

    return df


def test_regime_signals(df):
    """
    ==========================================================================
    MODULE 3: Regime Signal Generation
    ==========================================================================

    Purpose: Generate trading signals based on different regime detection methods.

    Sub-modules:
    - RegimeBO: Breakout and Turtle Trader signals
    - TripleMACrossoverRegime: Moving average crossover signals
    - RegimeFC: Floor/Ceiling regime detection

    Signal Values:
    - 1: Bullish / Long
    - 0: Neutral / Flat
    - -1: Bearish / Short
    """
    print("\n" + "=" * 70)
    print("MODULE 3: Regime Signal Generation")
    print("=" * 70)

    # =====================================================================
    # 3A: RegimeBO - Breakout and Turtle Trader Signals
    # =====================================================================
    print("\n--- 3A: RegimeBO (Breakout & Turtle Trader) ---")

    from algoshort.regime_bo import RegimeBO

    # Initialize with OHLC data
    regime_bo = RegimeBO(ohlc_stock=df)

    # ---------------------------------------------------------------------
    # Breakout Signal
    # ---------------------------------------------------------------------
    # Triggers when price breaks above highest high (bullish) or
    # below lowest low (bearish) over the window period
    print("\n1. Computing Breakout signals...")

    for window in [20, 50, 100]:
        df = regime_bo.compute_regime(
            regime_type='breakout',  # Single window breakout
            window=window,           # Lookback period
            relative=True,           # Use relative prices (ropen, rhigh, etc.)
            inplace=True             # Modify df in place
        )
        print(f"   Created: rbo_{window} (values: {df[f'rbo_{window}'].unique()})")

    # ---------------------------------------------------------------------
    # Turtle Trader Signal
    # ---------------------------------------------------------------------
    # Dual-window system: Enter on slow window breakout, exit on fast window
    # More conservative than simple breakout
    print("\n2. Computing Turtle Trader signals...")

    # Define fast/slow window pairs
    tt_pairs = [(20, 50)]  # (fast_window, slow_window)

    for fast, slow in tt_pairs:
        df = regime_bo.compute_regime(
            regime_type='turtle',    # Dual window system
            window=slow,             # Slow (entry) window
            fast_window=fast,        # Fast (exit) window
            relative=True,
            inplace=True
        )
        print(f"   Created: rtt_{slow}{fast} (values: {df[f'rtt_{slow}{fast}'].unique()})")

    # =====================================================================
    # 3B: TripleMACrossoverRegime - Moving Average Signals
    # =====================================================================
    print("\n--- 3B: TripleMACrossoverRegime (MA Crossover) ---")

    from algoshort.regime_ma import TripleMACrossoverRegime

    # Initialize with OHLC data
    regime_ma = TripleMACrossoverRegime(ohlc_stock=df)

    # ---------------------------------------------------------------------
    # Triple MA Crossover
    # ---------------------------------------------------------------------
    # Uses three moving averages (short, medium, long)
    # Bullish when short > medium > long
    # Bearish when short < medium < long
    print("\n3. Computing MA Crossover signals...")

    # MA parameters
    ma_params = {'short': 50, 'medium': 100, 'long': 150}

    for ma_type in ['sma', 'ema']:  # Simple MA and Exponential MA
        df = regime_ma.compute_ma_regime(
            ma_type=ma_type,
            short_window=ma_params['short'],
            medium_window=ma_params['medium'],
            long_window=ma_params['long'],
            relative=True,
            inplace=True
        )
        signal_name = f"r{ma_type}_{ma_params['short']}{ma_params['medium']}{ma_params['long']}"
        print(f"   Created: {signal_name}")

    # =====================================================================
    # 3C: RegimeFC - Floor/Ceiling Detection
    # =====================================================================
    print("\n--- 3C: RegimeFC (Floor/Ceiling Swing Analysis) ---")

    from algoshort.regime_fc import RegimeFC

    # Initialize with OHLC data
    # log_level controls verbosity (WARNING = less output)
    regime_fc = RegimeFC(df=df, log_level=logging.WARNING)

    # ---------------------------------------------------------------------
    # Floor/Ceiling Regime
    # ---------------------------------------------------------------------
    # Sophisticated swing detection algorithm that identifies:
    # - Support (floor) and resistance (ceiling) levels
    # - Regime changes based on swing structure
    print("\n4. Computing Floor/Ceiling regime...")

    df = regime_fc.compute_regime(
        relative=True,       # Use relative prices
        lvl=3,               # Swing level (higher = smoother)
        vlty_n=63,           # Volatility lookback period
        threshold=0.05,      # Swing threshold
        dgt=3,               # Decimal places
        d_vol=1,             # Volume filter
        dist_pct=0.05,       # Distance percentage
        retrace_pct=0.05,    # Retracement percentage
        r_vol=1.0            # Relative volume
    )

    print(f"   Created: rrg (Floor/Ceiling regime)")
    if 'rrg' in df.columns:
        print(f"   Values: {df['rrg'].value_counts().to_dict()}")

    # =====================================================================
    # Summary of all signals
    # =====================================================================
    print("\n--- Signal Summary ---")

    # Get all signal columns
    signal_columns = [col for col in df.columns
                      if any(col.startswith(prefix)
                             for prefix in ['rbo_', 'rtt_', 'rsma_', 'rema_', 'rrg'])
                      and not any(kw in col for kw in ['short', 'medium', 'long', '_ch'])]

    # Remove 'rrg_ch' if present (it's a change indicator, not a signal)
    signal_columns = [x for x in signal_columns if x != "rrg_ch"]

    print(f"\nTotal signals generated: {len(signal_columns)}")
    for sig in signal_columns:
        value_counts = df[sig].value_counts().to_dict()
        print(f"   {sig}: {value_counts}")

    return df, signal_columns


def test_signal_combiner(df, signal_columns):
    """
    ==========================================================================
    MODULE 4: Signal Combination (Grid Search)
    ==========================================================================

    Purpose: Combine multiple signals using entry/exit/direction logic.

    Classes:
    - HybridSignalCombiner: Combines direction, entry, and exit signals
    - SignalGridSearch: Tests all combinations of entry/exit signals

    Logic:
    - Direction signal: Determines long/short bias (floor/ceiling)
    - Entry signal: Triggers position entry (breakout, MA cross)
    - Exit signal: Triggers position exit (can be same or different signal)
    """
    print("\n" + "=" * 70)
    print("MODULE 4: Signal Combination (Grid Search)")
    print("=" * 70)

    from algoshort.combiner import SignalGridSearch, HybridSignalCombiner

    # ---------------------------------------------------------------------
    # Filter signals for grid search
    # ---------------------------------------------------------------------
    # Exclude direction signal from entry/exit options
    direction_col = 'rrg'  # Floor/ceiling regime as direction filter
    entry_exit_signals = [x for x in signal_columns if x != direction_col]

    print(f"\n1. Direction signal: {direction_col}")
    print(f"   Entry/Exit signals available: {len(entry_exit_signals)}")
    print(f"   Signals: {entry_exit_signals}")

    # ---------------------------------------------------------------------
    # Single combination example with HybridSignalCombiner
    # ---------------------------------------------------------------------
    print("\n2. Single Combination Example (HybridSignalCombiner):")

    combiner = HybridSignalCombiner(
        direction_col=direction_col,    # Overall market direction
        entry_col=entry_exit_signals[0], # Entry trigger
        exit_col=entry_exit_signals[0],  # Exit trigger (same as entry)
        verbose=False                    # Set True for trade-by-trade output
    )

    # Combine signals
    test_df = df.copy()
    test_df = combiner.combine_signals(
        test_df,
        output_col='test_signal',
        allow_flips=True,            # Allow direct long-to-short flips
        require_regime_alignment=True # Entries must align with direction
    )

    # Add metadata
    test_df = combiner.add_signal_metadata(test_df, 'test_signal')

    # Get trade summary
    summary = combiner.get_trade_summary(test_df, 'test_signal')
    print(f"   Total trades: {summary['total_entries']}")
    print(f"   Long trades: {summary['entry_long_count']}, Short trades: {summary['entry_short_count']}")
    print(f"   Time in market: {100 - summary['flat_pct']:.1f}%")

    # ---------------------------------------------------------------------
    # Full Grid Search (all combinations)
    # ---------------------------------------------------------------------
    print("\n3. Full Grid Search (SignalGridSearch):")

    # Initialize grid search
    searcher = SignalGridSearch(
        df=df.copy(),
        available_signals=entry_exit_signals,
        direction_col=direction_col
    )

    # Run parallel grid search
    # This tests ALL combinations of entry and exit signals
    results = searcher.run_grid_search_parallel(
        allow_flips=True,              # Allow position flips
        require_regime_alignment=True,  # Require direction alignment
        n_jobs=-1,                      # Use all CPU cores
        backend='multiprocessing'       # Parallelization method
    )

    # Get results summary
    print(f"\n   Combinations tested: {len(results)}")
    print(f"   Successful: {results['success'].sum()}")

    # Show top combinations by trade count
    top_combos = results.nlargest(3, 'total_trades')[['combination_name', 'total_trades', 'long_pct', 'short_pct']]
    print("\n   Top 3 most active combinations:")
    print(top_combos.to_string(index=False))

    # Update df with all combined signals
    df = searcher.df

    return df, results, signal_columns


def test_returns_calculator(df, signal_columns):
    """
    ==========================================================================
    MODULE 5: Returns Calculator
    ==========================================================================

    Purpose: Calculate returns, P&L, and equity curves for trading signals.

    Key Features:
    - Daily price changes based on signal direction
    - Cumulative P&L tracking
    - Log returns for proper compounding
    - Parallel processing for multiple signals

    Output Columns (for each signal):
    - {signal}_chg1D: Daily price change
    - {signal}_PL_cum: Cumulative P&L
    - {signal}_returns: Percentage returns
    - {signal}_log_returns: Log returns
    - {signal}_cumul: Cumulative returns
    """
    print("\n" + "=" * 70)
    print("MODULE 5: Returns Calculator")
    print("=" * 70)

    from algoshort.returns import ReturnsCalculator

    # ---------------------------------------------------------------------
    # Initialize the calculator
    # ---------------------------------------------------------------------
    # Specify column names for flexibility
    calc = ReturnsCalculator(
        ohlc_stock=df,
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        relative_prefix="r"  # Prefix for relative columns
    )

    # ---------------------------------------------------------------------
    # Calculate returns for all signals
    # ---------------------------------------------------------------------
    print("\n1. Calculating returns for all signals...")

    # Use relative=True to calculate returns on relative prices
    # This isolates strategy performance from market movements
    df = calc.get_returns_multiple(
        df=df,
        signals=signal_columns,
        relative=True,   # Use relative prices (rclose)
        n_jobs=-1,       # Use all CPU cores
        verbose=False    # Set True for progress updates
    )

    # ---------------------------------------------------------------------
    # Show results
    # ---------------------------------------------------------------------
    return_cols = [col for col in df.columns if '_chg1D_fx' in col]
    print(f"\n   Return columns created: {len(return_cols)}")

    # Show sample returns for first signal
    sample_sig = signal_columns[0]
    print(f"\n2. Sample returns for '{sample_sig}':")
    return_cols_sample = [col for col in df.columns if col.startswith(f'{sample_sig}_')]
    print(f"   Columns: {return_cols_sample}")

    # Calculate cumulative return
    if f'{sample_sig}_cumul' in df.columns:
        total_return = df[f'{sample_sig}_cumul'].iloc[-1]
        print(f"   Total cumulative return: {total_return:.2%}")

    return df


def test_stop_loss_calculator(df, signal_columns):
    """
    ==========================================================================
    MODULE 6: Stop Loss Calculator
    ==========================================================================

    Purpose: Calculate stop-loss levels for position risk management.

    Available Methods:
    - fixed_percentage: Fixed % below/above entry
    - atr: ATR-based (volatility-adjusted)
    - breakout_channel: Swing high/low based
    - moving_average: MA-based stops
    - volatility_std: Standard deviation based
    - support_resistance: Key level based
    - classified_pivot: Advanced pivot-based stops

    Stop Loss Logic:
    - Long position: Stop below current price
    - Short position: Stop above current price
    """
    print("\n" + "=" * 70)
    print("MODULE 6: Stop Loss Calculator")
    print("=" * 70)

    from algoshort.stop_loss import StopLossCalculator

    # ---------------------------------------------------------------------
    # Initialize the calculator
    # ---------------------------------------------------------------------
    calc = StopLossCalculator(df)

    # ---------------------------------------------------------------------
    # Calculate ATR-based stop losses for all signals
    # ---------------------------------------------------------------------
    print("\n1. Calculating ATR-based stop losses...")

    for signal in signal_columns:
        df = calc.atr_stop_loss(
            signal=signal,
            window=14,       # ATR lookback period
            multiplier=2.0   # ATR multiplier (higher = wider stops)
        )
        # Update calculator's data reference for next iteration
        calc.data = df

    # ---------------------------------------------------------------------
    # Demonstrate other stop loss methods
    # ---------------------------------------------------------------------
    print("\n2. Other Stop Loss Methods Available:")

    # Fixed percentage stop (example)
    sample_sig = signal_columns[0]

    print(f"\n   a) Fixed Percentage (5%):")
    test_df = calc.fixed_percentage_stop_loss(
        signal=sample_sig,
        percentage=0.05  # 5% stop
    )
    print(f"      Created: {sample_sig}_stop_loss")

    print(f"\n   b) Breakout Channel (20-day):")
    test_df = calc.breakout_channel_stop_loss(
        signal=sample_sig,
        window=20  # 20-day high/low
    )

    print(f"\n   c) Moving Average Stop (50-day):")
    test_df = calc.moving_average_stop_loss(
        signal=sample_sig,
        window=50,
        offset=0.0  # Optional offset from MA
    )

    print(f"\n   d) Volatility (Std Dev) Stop:")
    test_df = calc.volatility_std_stop_loss(
        signal=sample_sig,
        window=20,
        multiplier=1.5  # 1.5 standard deviations
    )

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    sl_cols = [col for col in df.columns if col.endswith('_stop_loss')]
    print(f"\n3. Stop loss columns created: {len(sl_cols)}")

    return df


def test_position_sizing(df, signal_columns):
    """
    ==========================================================================
    MODULE 7: Position Sizing
    ==========================================================================

    Purpose: Calculate optimal position sizes based on risk management.

    Position Sizing Strategies:
    - equal_weight: Fixed percentage of capital per trade
    - constant: Fixed risk amount per trade
    - concave: Conservative - reduce risk as drawdown increases
    - convex: Aggressive - increase risk when winning

    Key Concepts:
    - Risk is controlled via stop-loss distance
    - Position size = Risk Amount / (Entry - Stop Loss)
    - Different strategies adjust risk based on equity curve
    """
    print("\n" + "=" * 70)
    print("MODULE 7: Position Sizing")
    print("=" * 70)

    from algoshort.position_sizing import PositionSizing, run_position_sizing_parallel

    # ---------------------------------------------------------------------
    # Initialize the position sizer
    # ---------------------------------------------------------------------
    sizer = PositionSizing(
        tolerance=-0.10,      # Max drawdown tolerance (10%)
        mn=-0.0025,           # Minimum risk per trade (0.25%)
        mx=-0.05,             # Maximum risk per trade (5%)
        equal_weight=0.05,    # Equal weight allocation (5% of capital)
        avg=0.03,             # Average risk for constant strategy (3%)
        lot=1,                # Lot size (1 share minimum)
        initial_capital=100000 # Starting capital
    )

    # ---------------------------------------------------------------------
    # Calculate position sizes for all signals in parallel
    # ---------------------------------------------------------------------
    print("\n1. Calculating position sizes for all signals...")

    df = run_position_sizing_parallel(
        sizer=sizer,
        df=df,
        signals=signal_columns,
        chg_suffix="_chg1D_fx",    # Daily change column suffix
        sl_suffix="_stop_loss",    # Stop loss column suffix
        close_col='close',
        n_jobs=-1,                 # Use all CPU cores
        verbose=5                  # Progress verbosity
    )

    # ---------------------------------------------------------------------
    # Show results
    # ---------------------------------------------------------------------
    print("\n2. Equity curve columns created:")
    equity_cols = [col for col in df.columns if 'equity' in col]
    print(f"   Total equity columns: {len(equity_cols)}")

    # Sample equity statistics
    sample_equity = [col for col in equity_cols if 'equal' in col][:3]
    if sample_equity:
        print("\n3. Sample equity curve statistics:")
        print(df[sample_equity].describe().round(2))

    return df


def test_wrappers(df, signal_columns):
    """
    ==========================================================================
    MODULE 8: High-Level Wrappers
    ==========================================================================

    Purpose: Simplified functions that combine multiple steps.

    Available Wrappers:
    - generate_signals(): Generate all regime signals at once
    - calculate_return(): Calculate returns for signals
    - calculate_sl_signals(): Calculate stop losses
    - calculate_trading_edge(): Calculate risk metrics
    - calculate_risk_metrics(): Calculate risk metrics

    These functions are useful for:
    - Quick prototyping
    - Standardized workflows
    - Reduced boilerplate code
    """
    print("\n" + "=" * 70)
    print("MODULE 8: High-Level Wrappers")
    print("=" * 70)

    from algoshort.wrappers import (
        generate_signals,
        multiple_bo_signals,
        multiple_tt_signals,
        multiple_ma_signals,
        calculate_sl_signals
    )

    print("\n1. Available wrapper functions:")
    print("   - generate_signals(): Generate all regime signals")
    print("   - multiple_bo_signals(): Multiple breakout signals")
    print("   - multiple_tt_signals(): Multiple turtle trader signals")
    print("   - multiple_ma_signals(): Multiple MA crossover signals")
    print("   - calculate_sl_signals(): Calculate stop losses")
    print("   - calculate_return(): Calculate returns")
    print("   - calculate_trading_edge(): Calculate trading metrics")

    print("\n2. Example: generate_signals() combines all signal generators:")
    print("""
    df, signals = generate_signals(
        df=df,
        tt_search_space={'fast': [20], 'slow': [50]},
        bo_search_space=[100],
        ma_search_space={'short_ma': [50], 'medium_ma': [100], 'long_ma': [150]},
        relative=True
    )
    """)


def print_final_summary(df):
    """
    ==========================================================================
    FINAL SUMMARY
    ==========================================================================

    Display a summary of all columns created throughout the workflow.
    """
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n1. DataFrame shape: {df.shape}")

    # Categorize columns
    ohlc_cols = [c for c in df.columns if c in ['date', 'open', 'high', 'low', 'close', 'fx']]
    rel_ohlc = [c for c in df.columns if c.startswith('r') and c[1:] in ['open', 'high', 'low', 'close']]
    signal_cols = [c for c in df.columns if any(c.startswith(p) for p in ['rbo_', 'rtt_', 'rsma_', 'rema_', 'rrg'])
                   and not any(k in c for k in ['_chg', '_PL', '_returns', '_cumul', '_stop', '_equity', '_shares', '_risk', 'short', 'medium', 'long', '_ch'])]
    return_cols = [c for c in df.columns if any(k in c for k in ['_chg1D', '_PL_cum', '_returns', '_cumul'])]
    sl_cols = [c for c in df.columns if '_stop_loss' in c]
    equity_cols = [c for c in df.columns if '_equity_' in c]

    print(f"\n2. Column breakdown:")
    print(f"   OHLC columns: {len(ohlc_cols)}")
    print(f"   Relative OHLC: {len(rel_ohlc)}")
    print(f"   Signal columns: {len(signal_cols)}")
    print(f"   Return columns: {len(return_cols)}")
    print(f"   Stop loss columns: {len(sl_cols)}")
    print(f"   Equity columns: {len(equity_cols)}")
    print(f"   Other columns: {len(df.columns) - len(ohlc_cols) - len(rel_ohlc) - len(signal_cols) - len(return_cols) - len(sl_cols) - len(equity_cols)}")

    print("\n3. Memory usage:")
    mem_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"   Total: {mem_usage:.2f} MB")


def main():
    """
    ==========================================================================
    MAIN WORKFLOW
    ==========================================================================

    Executes the complete algoshort workflow step by step.
    """
    print("\n" + "=" * 70)
    print("ALGOSHORT LIBRARY - COMPREHENSIVE TEST")
    print("=" * 70)
    print("\nThis script demonstrates all modules of the algoshort library.")
    print("Each module is documented with its purpose and key methods.\n")

    try:
        # Step 1: Download data
        handler, df, bmk = test_yfinance_handler()

        # Step 2: Calculate relative prices
        df = test_ohlc_processor(df, bmk)

        # Step 3: Generate regime signals
        df, signal_columns = test_regime_signals(df)

        # Step 4: Combine signals (grid search)
        df, results, signal_columns = test_signal_combiner(df, signal_columns)

        # Step 5: Calculate returns
        df = test_returns_calculator(df, signal_columns)

        # Step 6: Calculate stop losses
        df = test_stop_loss_calculator(df, signal_columns)

        # Step 7: Position sizing
        df = test_position_sizing(df, signal_columns)

        # Step 8: Show wrappers
        test_wrappers(df, signal_columns)

        # Final summary
        print_final_summary(df)

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        return df, results

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    df, results = main()
