"""
Score Research Dataset with Leader_Score_V2

Loads the research dataset, applies Leader_Score_V2 (IC-validated weights),
and saves a scored CSV ready for backtesting or further analysis.

Leader_Score_V2 weights (IC-validated on 204,105 stock-days):
    RS10_Pct              0.40
    RS5_Pct               0.25
    RVOL_Pct              0.15
    VolatilityContraction 0.10  (reverse-ranked: lower=better)
    High52_Proximity      0.05  (higher=better)
    RS3_Pct               0.05
    RS2_Pct               DROPPED
    SPY_Bull gate         zeroes score on non-bull days

Usage
-----
    python leader_score_v2/score_research_dataset.py

    python leader_score_v2/score_research_dataset.py \\
        --input  leader_score_v2/output/research_dataset.csv \\
        --output leader_score_v2/output/scored_dataset.csv \\
        --no-regime-gate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.leader_score_v2 import compute_leader_score_v2, ALL_WEIGHTS

DEFAULT_INPUT  = str(ROOT / "output" / "research_dataset.csv")
DEFAULT_OUTPUT = str(ROOT / "output" / "scored_dataset.csv")


def _main():
    parser = argparse.ArgumentParser(
        description="Apply Leader_Score_V2 to research dataset"
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Input research_dataset.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output scored_dataset.csv")
    parser.add_argument("--no-regime-gate", action="store_true",
                        help="Disable SPY regime gate (score all days)")
    parser.add_argument("--threshold", type=float, default=80.0,
                        help="Leader threshold for summary stats (default 80)")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path, parse_dates=["Date"], low_memory=False)
    print(f"Loaded : {len(df):,} rows x {df.shape[1]} columns")
    print(f"Tickers: {df['Ticker'].nunique()} | Dates: {df['Date'].nunique()}")

    print(f"\nApplying Leader_Score_V2 ...")
    print(f"Weights: {ALL_WEIGHTS}")
    print(f"SPY regime gate: {'OFF' if args.no_regime_gate else 'ON'}")

    df = compute_leader_score_v2(
        df,
        apply_regime_gate=not args.no_regime_gate,
    )

    # Summary stats
    score_col = "Leader_Score_V2"
    print(f"\nScore distribution:")
    print(f"  Min    : {df[score_col].min():.2f}")
    print(f"  25th   : {df[score_col].quantile(0.25):.2f}")
    print(f"  Median : {df[score_col].median():.2f}")
    print(f"  75th   : {df[score_col].quantile(0.75):.2f}")
    print(f"  90th   : {df[score_col].quantile(0.90):.2f}")
    print(f"  Max    : {df[score_col].max():.2f}")

    n_leaders = (df[score_col] >= args.threshold).sum()
    pct = n_leaders / len(df) * 100
    print(f"\nLeaders (score >= {args.threshold}): {n_leaders:,} rows ({pct:.1f}%)")

    # Top leaders per date (sample)
    top_leaders = (
        df[df[score_col] >= args.threshold]
        .sort_values(["Date", score_col], ascending=[True, False])
    )
    print(f"\nSample top leaders:")
    print(top_leaders[["Date", "Ticker", "Sector", score_col,
                        "RS10_Pct", "RS5_Pct", "RVOL_Pct",
                        "DollarVolume_Pct", "RS3_Pct"]].head(15).to_string(index=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    _main()
