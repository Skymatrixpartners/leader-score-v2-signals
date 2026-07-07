"""
Structure factors: proximity to highs, breakout quality, base analysis.
"""
import pandas as pd
import numpy as np


def compute_high_proximity(
    closes: pd.Series,
    highs: pd.Series,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Measure how close price is to its N-day high (default 52-week).

    High52           = Rolling max of daily High over `lookback` days
    High52_Proximity = Close / High52

    Values near 1.0 indicate the stock is at or near a yearly high
    (prime breakout territory).

    Args:
        closes:   Series of daily Close prices.
        highs:    Series of daily High prices.
        lookback: Rolling window in trading days (default 252 = 1 year).

    Returns:
        DataFrame with columns: High52, High52_Proximity
    """
    high52 = highs.rolling(lookback, min_periods=1).max()
    proximity = closes / high52.replace(0, np.nan)

    return pd.DataFrame({"High52": high52, "High52_Proximity": proximity})


def compute_clv(
    closes: pd.Series,
    lows: pd.Series,
    highs: pd.Series,
) -> pd.DataFrame:
    """
    Close Location Value (CLV): where the close sits within the daily range.

    CLV = (Close - Low) / (High - Low)

    1.0 = closed exactly at the high  (strong buyer control)
    0.5 = closed at midpoint
    0.0 = closed exactly at the low   (seller control)

    Doji bars where High == Low are set to NaN (no valid range).

    Args:
        closes: Series of daily Close prices.
        lows:   Series of daily Low prices.
        highs:  Series of daily High prices.

    Returns:
        DataFrame with column: Daily_CLV
    """
    range_ = (highs - lows).replace(0, np.nan)
    clv = (closes - lows) / range_

    return pd.DataFrame({"Daily_CLV": clv})
