"""
Volume factors: RVOL, volume surge, dollar volume, volume confirmation.
"""
import pandas as pd
import numpy as np


def compute_rvol(
    volumes: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    Relative Volume: today's volume vs rolling average.

    AvgVol20D = rolling mean of volume over `window` days
    RVOL      = Volume / AvgVol20D

    Values > 1.0 indicate above-average participation.

    Args:
        volumes: Series of daily traded volume.
        window:  Lookback for average volume (default 20).

    Returns:
        DataFrame with columns: AvgVol20D, RVOL
    """
    avg_vol = volumes.rolling(window).mean()
    rvol = volumes / avg_vol.replace(0, np.nan)

    return pd.DataFrame({"AvgVol20D": avg_vol, "RVOL": rvol})


def compute_dollar_volume(
    closes: pd.Series,
    volumes: pd.Series,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Dollar Volume: institutional-grade liquidity measure.

    DollarVolume   = Close × Volume
    DollarVolume20 = rolling mean of DollarVolume over `lookback` days

    Higher dollar volume indicates institutional-level tradability.

    Args:
        closes:   Series of daily Close prices.
        volumes:  Series of daily traded volume.
        lookback: Rolling average period (default 20).

    Returns:
        DataFrame with columns: DollarVolume, DollarVolume20
    """
    dv = closes * volumes
    dv20 = dv.rolling(lookback).mean()

    return pd.DataFrame({"DollarVolume": dv, "DollarVolume20": dv20})
