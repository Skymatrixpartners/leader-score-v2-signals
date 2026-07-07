"""
Volatility factors: ATR, range compression, volatility contraction.
"""
import pandas as pd
import numpy as np


def compute_atr(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 20,
) -> pd.DataFrame:
    """
    Compute Average True Range (ATR) and related expansion metrics.

    True Range = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
    ATR20      = rolling mean of True Range over `period` days
    Today_Range  = High - Low (simple intraday range)
    ATR_Expansion = Today_Range / ATR20

    Args:
        highs:   Series of daily High prices.
        lows:    Series of daily Low prices.
        closes:  Series of daily Close prices.
        period:  ATR lookback window (default 20).

    Returns:
        DataFrame with columns: ATR20, Today_Range, ATR_Expansion
    """
    prev_close = closes.shift(1)

    tr = pd.concat(
        [
            highs - lows,
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr20 = tr.rolling(period).mean()
    today_range = highs - lows
    atr_expansion = today_range / atr20.replace(0, np.nan)

    return pd.DataFrame(
        {"ATR20": atr20, "Today_Range": today_range, "ATR_Expansion": atr_expansion}
    )


def compute_volatility_contraction(
    highs: pd.Series,
    lows: pd.Series,
    short_window: int = 5,
    long_window: int = 20,
) -> pd.DataFrame:
    """
    Detect volatility contraction (tight base before breakout).

    DailyRange          = High - Low
    AvgRange5           = short_window-day average of DailyRange
    AvgRange20          = long_window-day average of DailyRange
    VolatilityContraction = AvgRange5 / AvgRange20

    Values < 1.0 indicate contraction (range tightening = potential base).

    Args:
        highs:        Series of daily High prices.
        lows:         Series of daily Low prices.
        short_window: Short lookback for range average (default 5).
        long_window:  Long lookback for range average (default 20).

    Returns:
        DataFrame with columns: AvgRange5, AvgRange20, VolatilityContraction
    """
    daily_range = highs - lows
    avg_short = daily_range.rolling(short_window).mean()
    avg_long = daily_range.rolling(long_window).mean()
    contraction = avg_short / avg_long.replace(0, np.nan)

    return pd.DataFrame(
        {
            "AvgRange5": avg_short,
            "AvgRange20": avg_long,
            "VolatilityContraction": contraction,
        }
    )
