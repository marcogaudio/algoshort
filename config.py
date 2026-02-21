"""
Central configuration for algoshort trading analysis.

All parameters used in generate_notebooks.py and analyze_results.py
are defined here to ensure consistency across the system.
"""

# =============================================================================
# MARKET CONFIGURATIONS
# =============================================================================
MARKETS = {
    "italy": {
        "name": "Italian Stock Market (FTSE MIB)",
        "benchmark": "FTSEMIB.MI",
        "tickers_file": "tickers_italy.txt",
        "output_dir": "notebooks/italy",
        "results_dir": "results/italy",
    },
    "nasdaq": {
        "name": "NASDAQ (Top 100)",
        "benchmark": "^IXIC",
        "tickers_file": "tickers_nasdaq.txt",
        "output_dir": "notebooks/nasdaq",
        "results_dir": "results/nasdaq",
    },
}

DEFAULT_MARKET = "italy"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
START_DATE = "2024-01-01"
INITIAL_CAPITAL = 10000

# Legacy single benchmark (for backward compatibility)
BENCHMARK = MARKETS[DEFAULT_MARKET]["benchmark"]

# =============================================================================
# BREAKOUT STRATEGY (rbo_20, rbo_50, rbo_100)
# =============================================================================
BREAKOUT_WINDOWS = [20, 50, 100]

# =============================================================================
# TURTLE TRADER STRATEGY (rtt_5020)
# =============================================================================
TURTLE_ENTRY_WINDOW = 50
TURTLE_EXIT_WINDOW = 20

# =============================================================================
# TRIPLE MA CROSSOVER (rsma/rema_50100150)
# =============================================================================
MA_SHORT = 50
MA_MEDIUM = 100
MA_LONG = 150

# =============================================================================
# FLOOR/CEILING REGIME (rrg)
# =============================================================================
FC_LEVEL = 3
FC_VOLATILITY_WINDOW = 63
FC_THRESHOLD = 0.05
FC_RETRACEMENT = 0.05
FC_DGT = 3
FC_D_VOL = 1
FC_DIST_PCT = 0.05
FC_R_VOL = 1.0

# =============================================================================
# STOP LOSS (Wider stops for aggressive profile)
# =============================================================================
STOP_LOSS_ATR_WINDOW = 14
STOP_LOSS_ATR_MULTIPLIER = 2.5   # Wider stops - less whipsaw, more risk

# =============================================================================
# POSITION SIZING (Aggressive Profile)
# =============================================================================
POSITION_TOLERANCE = -0.35       # Max drawdown tolerance (35%) — must stay negative (drawdown divisor)
POSITION_MIN_RISK = 0.25         # Min risk per trade (25%)
POSITION_MAX_RISK = 0.60         # Max risk per trade (60%)
POSITION_EQUAL_WEIGHT = 0.10     # Equal weight per position (10%) - larger allocations
POSITION_AVG = 0.06              # Average position size (6%)
POSITION_LOT = 1                 # Lot size

# =============================================================================
# SCORING WEIGHTS (for ranking trades)
# =============================================================================
SCORE_WEIGHT_RETURN = 0.40       # Weight for total return
SCORE_WEIGHT_DRAWDOWN = 0.30     # Weight for max drawdown
SCORE_WEIGHT_RISK_ADJ = 0.30     # Weight for risk-adjusted return

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
TOP_N_TRADES = 10                # Number of top trades to show


def get_config_summary() -> str:
    """Return a formatted string with all configuration parameters."""
    return f"""
  Data Period:
    Start Date:       {START_DATE}
    Benchmark:        {BENCHMARK}

  Breakout Strategy (rbo_20, rbo_50, rbo_100):
    Windows:          {', '.join(map(str, BREAKOUT_WINDOWS))} days
    Logic:            LONG on N-day high breakout, SHORT on N-day low

  Turtle Trader (rtt_{TURTLE_ENTRY_WINDOW}{TURTLE_EXIT_WINDOW}):
    Entry Window:     {TURTLE_ENTRY_WINDOW} days (breakout)
    Exit Window:      {TURTLE_EXIT_WINDOW} days (reversal)

  Triple MA Crossover (rsma/rema_{MA_SHORT}{MA_MEDIUM}{MA_LONG}):
    Fast MA:          {MA_SHORT} days
    Medium MA:        {MA_MEDIUM} days
    Slow MA:          {MA_LONG} days
    Logic:            LONG when Fast > Medium > Slow

  Floor/Ceiling Regime (rrg):
    Swing Level:      {FC_LEVEL}
    Volatility Window: {FC_VOLATILITY_WINDOW} days
    Threshold:        {int(FC_THRESHOLD * 100)}%
    Retracement:      {int(FC_RETRACEMENT * 100)}%

  Stop Loss:
    Method:           ATR-based
    ATR Window:       {STOP_LOSS_ATR_WINDOW} days
    Multiplier:       {STOP_LOSS_ATR_MULTIPLIER}x ATR

  Position Sizing:
    Initial Capital:  {INITIAL_CAPITAL:,}
    Equal Weight:     {int(POSITION_EQUAL_WEIGHT * 100)}% per position
    Risk per Trade:   {int(abs(POSITION_MIN_RISK) * 100)}% - {int(abs(POSITION_MAX_RISK) * 100)}%
    Max Drawdown:     {int(abs(POSITION_TOLERANCE) * 100)}% tolerance

  Scoring Weights:
    Total Return:     {int(SCORE_WEIGHT_RETURN * 100)}%
    Max Drawdown:     {int(SCORE_WEIGHT_DRAWDOWN * 100)}%
    Risk-Adj Return:  {int(SCORE_WEIGHT_RISK_ADJ * 100)}%
"""
