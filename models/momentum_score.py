"""
Momentum score: IC + quintile validated composite for 1-10 day holds.

Final weights (all factors on 0-100 cross-sectional percentile scale):
    RS10_Pct          0.45  Perfect staircase, IC_IR=0.34 at 10D
    RS5_Pct           0.25  Mostly monotonic, IC_IR=0.20 at 10D
    RVOL_Pct          0.20  Mostly monotonic, IC_IR=0.25 at 10D
    RS3_Pct           0.05  Small diversification contribution
    DollarVolume_Pct  0.05  Q5 captures institutional liquidity surge
    ---------------------------------------------------------------
    Total             1.00

Dropped:
    RS2_Pct           -- Most noise, no IC edge
    VolatilityContraction -- U-shaped quintile
    High52_Proximity  -- Inverted quintile (near high = worse)
    DollarVolume20    -- 0.935 corr with DollarVolume (redundant)
"""
import pandas as pd
import numpy as np


# IC + quintile validated weights (sum = 1.00)
MOMENTUM_WEIGHTS = {
    "RS10_Pct":         0.45,
    "RS5_Pct":          0.25,
    "RVOL_Pct":         0.20,
    "RS3_Pct":          0.05,
    "DollarVolume_Pct": 0.05,
}


def compute_momentum_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute weighted momentum score for each row.

    All input columns must already be cross-sectional percentile
    ranks (0-100 scale, computed daily across the universe).
    DollarVolume_Pct must be added by add_dollar_volume_pct() before calling.

    Args:
        df: DataFrame with columns RS10_Pct, RS5_Pct, RVOL_Pct, RS3_Pct, DollarVolume_Pct.

    Returns:
        Series of scores on 0-100 scale (weights sum to 1.00).
    """
    missing = [c for c in MOMENTUM_WEIGHTS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing momentum factor columns: {missing}")

    score = pd.Series(0.0, index=df.index)
    for factor, weight in MOMENTUM_WEIGHTS.items():
        score += df[factor].fillna(50.0) * weight   # NaN -> neutral (50th pct)

    return score
