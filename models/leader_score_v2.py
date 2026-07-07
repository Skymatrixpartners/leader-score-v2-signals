"""
Leader_Score_V2 -- IC + Quintile Validated Composite Score

Final weights (all on 0-100 cross-sectional percentile scale):
    RS10_Pct         0.45  Perfect staircase, IC_IR=0.34 at 10D
    RS5_Pct          0.25  Mostly monotonic, IC_IR=0.20 at 10D
    RVOL_Pct         0.20  Mostly monotonic, IC_IR=0.25 at 10D
    RS3_Pct          0.05  Small diversification contribution
    DollarVolume_Pct 0.05  Q5 captures institutional liquidity surge
    ------------------------------------------------------------------
    Total            1.00

Dropped vs prior version:
    VolatilityContraction -- U-shaped quintile, not a clean linear predictor
    High52_Proximity      -- Inverted quintile (Q2/Q3 beat Q5, near-high = worse)
    DollarVolume20        -- 0.935 correlated with DollarVolume (redundant)

SPY Regime Gate:
    When SPY is not in a bull regime (Close > SMA50 > SMA200),
    Leader_Score_V2 is set to 0. Avoids fighting market downtrends.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.momentum_score import compute_momentum_score, MOMENTUM_WEIGHTS

# Expose for external use
ALL_WEIGHTS = MOMENTUM_WEIGHTS


# ---------------------------------------------------------------------------
# Per-date percentile rank for DollarVolume
# ---------------------------------------------------------------------------

def add_dollar_volume_pct(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Add cross-sectional percentile rank of DollarVolume per date.

    Higher DollarVolume -> higher rank (institutional liquidity = bullish).

    Args:
        df:       Research DataFrame with DollarVolume column.
        date_col: Date column for grouping.

    Returns:
        DataFrame with DollarVolume_Pct column added.
    """
    if "DollarVolume" not in df.columns:
        raise ValueError("DollarVolume column not found.")

    df = df.copy()
    df["DollarVolume_Pct"] = (
        df.groupby(date_col)["DollarVolume"]
        .rank(pct=True, ascending=True)
        * 100
    )
    return df


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def compute_leader_score_v2(
    df: pd.DataFrame,
    apply_regime_gate: bool = True,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Compute Leader_Score_V2 on the full research dataset.

    Steps:
        1. Add DollarVolume_Pct (per-date percentile rank).
        2. Compute weighted composite score from all 5 validated factors.
        3. Optionally zero out scores when SPY is not in bull regime.

    Required columns in df:
        RS10_Pct, RS5_Pct, RVOL_Pct, RS3_Pct, DollarVolume,
        Date (for groupby), SPY_Bull (for regime gate)

    Args:
        df:                Research DataFrame.
        apply_regime_gate: Zero out score on non-bull SPY days (default True).
        date_col:          Date column name.

    Returns:
        DataFrame with DollarVolume_Pct and Leader_Score_V2 columns added.
    """
    required = ["RS10_Pct", "RS5_Pct", "RVOL_Pct", "RS3_Pct", "DollarVolume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Step 1 -- Per-date percentile rank for DollarVolume
    df = add_dollar_volume_pct(df, date_col=date_col)

    # Step 2 -- Weighted composite score (all factors already 0-100 pct rank)
    df["Leader_Score_V2"] = compute_momentum_score(df)

    # Step 3 -- SPY regime gate
    if apply_regime_gate and "SPY_Bull" in df.columns:
        non_bull = ~df["SPY_Bull"].fillna(False)
        n_gated  = non_bull.sum()
        df.loc[non_bull, "Leader_Score_V2"] = 0.0
        print(f"[leader_score_v2] SPY gate: {n_gated:,} rows zeroed "
              f"({n_gated / len(df) * 100:.1f}% of dataset)")

    return df
