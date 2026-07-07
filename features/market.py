"""
Market/regime factors: SPY trend, breadth, risk-on/risk-off gates.
"""
import pandas as pd
import numpy as np


def compute_spy_regime(
    spy_closes: pd.Series,
    short_ma: int = 50,
    long_ma: int = 200,
) -> pd.DataFrame:
    """
    Classify the SPY market regime using SMA crossover.

    SPY_SMA50  = 50-day simple moving average of SPY Close
    SPY_SMA200 = 200-day simple moving average of SPY Close

    SPY_Bull    = Close > SMA50  AND  SMA50 > SMA200  (clear uptrend)
    SPY_Bear    = Close < SMA50  AND  SMA50 < SMA200  (clear downtrend)
    SPY_Neutral = all other combinations (transitioning / conflicting)

    Args:
        spy_closes: Series of SPY daily Close prices (DatetimeIndex recommended).
        short_ma:   Short SMA period (default 50).
        long_ma:    Long SMA period (default 200).

    Returns:
        DataFrame with columns:
            SPY_Close, SPY_SMA50, SPY_SMA200,
            SPY_Bull (bool), SPY_Bear (bool), SPY_Neutral (bool)
    """
    sma50 = spy_closes.rolling(short_ma).mean()
    sma200 = spy_closes.rolling(long_ma).mean()

    bull = (spy_closes > sma50) & (sma50 > sma200)
    bear = (spy_closes < sma50) & (sma50 < sma200)
    neutral = ~bull & ~bear

    return pd.DataFrame(
        {
            "SPY_Close": spy_closes,
            "SPY_SMA50": sma50,
            "SPY_SMA200": sma200,
            "SPY_Bull": bull,
            "SPY_Bear": bear,
            "SPY_Neutral": neutral,
        }
    )


def compute_market_regime_gate(
    spy_closes: pd.Series,
    period: int = 200,
) -> pd.Series:
    """
    Simple binary entry gate: True if SPY is above its long-term MA.

    Args:
        spy_closes: Series of SPY daily Close prices.
        period:     Long-term MA period (default 200).

    Returns:
        Boolean Series — True means regime allows long entries.
    """
    sma = spy_closes.rolling(period).mean()
    return spy_closes > sma
