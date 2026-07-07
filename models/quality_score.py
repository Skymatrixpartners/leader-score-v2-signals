"""
Quality Feature Engineering for Leader Score V2.

This module computes all structural and quality-based features required
for IC analysis, quintile research, and eventual construction of
Leader Score V2.  No composite score is produced here — every feature
is stored as its own raw column for downstream research.

Feature Groups
--------------
1.  ATR Expansion          ATR20, Today_Range, ATR_Expansion
2.  52-Week High Proximity High52, High52_Proximity
3.  Volatility Contraction AvgRange5, AvgRange20, VolatilityContraction
4.  Close Location Value   Daily_CLV
5.  Dollar Volume          DollarVolume, DollarVolume20
6.  Gap Extension Risk     GapPct, GapExtensionATR
7.  SPY Market Regime      SPY_Close, SPY_SMA50, SPY_SMA200,
                           SPY_Bull, SPY_Bear, SPY_Neutral
8.  Earnings Flags         DaysSinceEarnings, DaysUntilNextEarnings,
                           RecentEarningsFlag, UpcomingEarningsFlag

Usage
-----
    from models.quality_score import build_quality_features

    research_df = build_quality_features(
        stock_df=daily_ohlcv,   # Date-indexed OHLCV for one stock
        spy_df=spy_daily,       # Date-indexed SPY OHLCV
    )
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow sibling-package imports when running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.volatility import compute_atr, compute_volatility_contraction
from features.structure import compute_high_proximity, compute_clv
from features.volume import compute_rvol, compute_dollar_volume
from features.market import compute_spy_regime
from features.events import compute_earnings_flags


# ---------------------------------------------------------------------------
# 1.  ATR Expansion
# ---------------------------------------------------------------------------

def add_atr_expansion(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    """
    Append ATR expansion features.

    Added columns: ATR20, Today_Range, ATR_Expansion

    Args:
        df:     Daily OHLCV DataFrame with columns High, Low, Close.
        period: ATR lookback window (default 20).

    Returns:
        Copy of df with three new columns appended.
    """
    atr_df = compute_atr(df["High"], df["Low"], df["Close"], period=period)
    return pd.concat([df, atr_df], axis=1)


# ---------------------------------------------------------------------------
# 2.  52-Week High Proximity
# ---------------------------------------------------------------------------

def add_high_proximity(
    df: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Append 52-week high proximity features.

    Added columns: High52, High52_Proximity

    Args:
        df:       Daily OHLCV DataFrame with columns High, Close.
        lookback: Rolling window in trading days (default 252).

    Returns:
        Copy of df with two new columns appended.
    """
    prox_df = compute_high_proximity(df["Close"], df["High"], lookback=lookback)
    return pd.concat([df, prox_df], axis=1)


# ---------------------------------------------------------------------------
# 3.  Volatility Contraction
# ---------------------------------------------------------------------------

def add_volatility_contraction(
    df: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
) -> pd.DataFrame:
    """
    Append volatility contraction features.

    Added columns: AvgRange5, AvgRange20, VolatilityContraction

    Args:
        df:           Daily OHLCV DataFrame with columns High, Low.
        short_window: Short range-average period (default 5).
        long_window:  Long range-average period (default 20).

    Returns:
        Copy of df with three new columns appended.
    """
    vcp_df = compute_volatility_contraction(
        df["High"], df["Low"], short_window, long_window
    )
    return pd.concat([df, vcp_df], axis=1)


# ---------------------------------------------------------------------------
# 4.  Close Location Value
# ---------------------------------------------------------------------------

def add_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append daily Close Location Value.

    Added column: Daily_CLV

    CLV = (Close - Low) / (High - Low).
    1.0 = closed at high; 0.0 = closed at low.

    Args:
        df: Daily OHLCV DataFrame with columns High, Low, Close.

    Returns:
        Copy of df with Daily_CLV appended.
    """
    clv_df = compute_clv(df["Close"], df["Low"], df["High"])
    return pd.concat([df, clv_df], axis=1)


# ---------------------------------------------------------------------------
# 5.  Dollar Volume
# ---------------------------------------------------------------------------

def add_dollar_volume(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Append dollar volume features.

    Added columns: DollarVolume, DollarVolume20

    Args:
        df:       Daily OHLCV DataFrame with columns Close, Volume.
        lookback: Rolling average period (default 20).

    Returns:
        Copy of df with two new columns appended.
    """
    dv_df = compute_dollar_volume(df["Close"], df["Volume"], lookback=lookback)
    return pd.concat([df, dv_df], axis=1)


# ---------------------------------------------------------------------------
# 6.  Gap Extension Risk
# ---------------------------------------------------------------------------

def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append gap extension risk features.

    GapPct          = (Open - PrevClose) / PrevClose
    GapExtensionATR = |GapPct| * Close / ATR20
                      (gap size expressed in units of ATR)

    NOTE: Requires ATR20 column — call add_atr_expansion() first.

    Added columns: GapPct, GapExtensionATR

    Args:
        df: Daily OHLCV DataFrame that already contains ATR20, Open, Close.

    Returns:
        Copy of df with two new columns appended.
    """
    prev_close = df["Close"].shift(1)
    gap_pct = (df["Open"] - prev_close) / prev_close.replace(0, np.nan)

    if "ATR20" in df.columns:
        gap_ext_atr = gap_pct.abs() * df["Close"] / df["ATR20"].replace(0, np.nan)
    else:
        gap_ext_atr = pd.Series(np.nan, index=df.index)

    return df.assign(GapPct=gap_pct, GapExtensionATR=gap_ext_atr)


# ---------------------------------------------------------------------------
# 7.  SPY Market Regime
# ---------------------------------------------------------------------------

def add_spy_regime(
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge SPY market regime features into the stock DataFrame.

    Added columns: SPY_Close, SPY_SMA50, SPY_SMA200,
                   SPY_Bull, SPY_Bear, SPY_Neutral

    Args:
        df:      Stock daily DataFrame (Date as index).
        spy_df:  SPY daily DataFrame with column Close (Date as index).

    Returns:
        Stock DataFrame left-merged with SPY regime columns on the date index.
    """
    regime_df = compute_spy_regime(spy_df["Close"])
    regime_df.index = spy_df.index
    return df.merge(regime_df, left_index=True, right_index=True, how="left")


# ---------------------------------------------------------------------------
# 8.  Earnings Flags
# ---------------------------------------------------------------------------

def add_earnings_flags(
    df: pd.DataFrame,
    earnings_dates: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    Append earnings proximity flag columns.

    Added columns: DaysSinceEarnings, DaysUntilNextEarnings,
                   RecentEarningsFlag, UpcomingEarningsFlag

    If no earnings_dates are provided all columns are NaN (placeholder
    until an earnings API is integrated).

    Args:
        df:             Stock daily DataFrame (Date as index).
        earnings_dates: DatetimeIndex of earnings announcement dates (optional).

    Returns:
        Copy of df with four earnings columns appended.
    """
    earnings_df = compute_earnings_flags(df.index, earnings_dates)
    return pd.concat([df, earnings_df], axis=1)


# ---------------------------------------------------------------------------
# Master research dataset builder
# ---------------------------------------------------------------------------

def build_quality_features(
    stock_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    earnings_dates: pd.DatetimeIndex = None,
    atr_period: int = 20,
    high_lookback: int = 252,
) -> pd.DataFrame:
    """
    Build the complete quality feature dataset for a single stock.

    Applies all eight feature groups in sequence and returns a single
    DataFrame where every feature is its own column.  No composite
    score is computed — this output is intended for IC analysis,
    quintile research, and correlation checks.

    Pipeline order
    --------------
    1. ATR Expansion           (must run before gap features)
    2. 52-Week High Proximity
    3. Volatility Contraction
    4. Close Location Value
    5. Dollar Volume
    6. Gap Extension Risk      (requires ATR20 from step 1)
    7. SPY Market Regime
    8. Earnings Flags

    Required columns in stock_df
    -----------------------------
    Date (as index or column), Open, High, Low, Close, Volume

    Required columns in spy_df
    --------------------------
    Date (as index or column), Close

    Args:
        stock_df:       Daily OHLCV DataFrame for a single stock.
        spy_df:         Daily SPY DataFrame aligned to the same date range.
        earnings_dates: Optional DatetimeIndex of earnings dates.
        atr_period:     ATR lookback window (default 20).
        high_lookback:  52-week high rolling window (default 252).

    Returns:
        DataFrame indexed by Date with original OHLCV plus all
        quality feature columns.
    """
    # --- Normalise index ---
    df = stock_df.copy()
    if "Date" in df.columns:
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    spy = spy_df.copy()
    if "Date" in spy.columns:
        spy = spy.set_index("Date")
    spy.index = pd.to_datetime(spy.index)

    # Step 1 — ATR Expansion (must precede gap features)
    df = add_atr_expansion(df, period=atr_period)

    # Step 2 — 52-Week High Proximity
    df = add_high_proximity(df, lookback=high_lookback)

    # Step 3 — Volatility Contraction
    df = add_volatility_contraction(df)

    # Step 4 — Close Location Value
    df = add_clv(df)

    # Step 5 — Dollar Volume
    df = add_dollar_volume(df)

    # Step 6 — Gap Extension Risk (needs ATR20)
    df = add_gap_features(df)

    # Step 7 — SPY Market Regime
    df = add_spy_regime(df, spy)

    # Step 8 — Earnings Flags (placeholders if no data)
    df = add_earnings_flags(df, earnings_dates)

    return df
