"""
Build Research Dataset â€” Leader Score V2

Connects the momentum feature pipeline (features/momentum.py) with the
quality feature pipeline (models/quality_score.py) to produce one master
research CSV containing every feature for every ticker on every trading day.

Output columns
--------------
Identifiers:
    Date, Ticker, Sector

Raw momentum values:
    Close, Volume, AvgVol20D, RVOL,
    RS2, RS3, RS5, RS10

Cross-sectional percentile ranks (0â€“100, computed daily across universe):
    RS2_Pct, RS3_Pct, RS5_Pct, RS10_Pct, RVOL_Pct

Quality features (raw, no scoring):
    ATR20, Today_Range, ATR_Expansion
    High52, High52_Proximity
    AvgRange5, AvgRange20, VolatilityContraction
    Daily_CLV
    DollarVolume, DollarVolume20
    GapPct, GapExtensionATR
    SPY_Close, SPY_SMA50, SPY_SMA200, SPY_Bull, SPY_Bear, SPY_Neutral
    DaysSinceEarnings, DaysUntilNextEarnings, RecentEarningsFlag, UpcomingEarningsFlag

Usage
-----
    python build_research_dataset.py

    # or with a custom universe / output path:
    python build_research_dataset.py \\
        --universe  project/data/russell_1000.csv \\
        --output    leader_score_v2/output/research_dataset.csv \\
        --lookback  365 \\
        --warmup    90
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from features/ and models/ regardless of cwd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from features.momentum import fetch_daily_bars, safe_rs
from features.events import load_earnings_calendar
from models.quality_score import build_quality_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile_rank(series: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank (0â€“100)."""
    return series.rank(pct=True) * 100


def compute_spy_returns(spy_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute rolling SPY returns used for RS calculations."""
    spy = spy_df.sort_values("Date").copy()
    spy["SPY_Ret2"]  = spy["Close"].pct_change(2)
    spy["SPY_Ret3"]  = spy["Close"].pct_change(3)
    spy["SPY_Ret5"]  = spy["Close"].pct_change(5)
    spy["SPY_Ret10"] = spy["Close"].pct_change(10)
    return spy


# ---------------------------------------------------------------------------
# Per-ticker processing
# ---------------------------------------------------------------------------

def process_ticker(
    ticker: str,
    sector: str,
    fetch_from: str,
    fetch_to: str,
    spy_returns: pd.DataFrame,
    earnings_calendar: dict = None,
) -> pd.DataFrame | None:
    """
    Fetch full OHLCV for one ticker, compute raw momentum factors and
    all quality features.  Returns a tidy DataFrame indexed by Date,
    or None if the ticker has insufficient data.

    Args:
        ticker:       Ticker symbol.
        sector:       Sector string from universe file.
        fetch_from:   Start date string YYYY-MM-DD (includes warmup).
        fetch_to:     End date string YYYY-MM-DD.
        spy_returns:  SPY DataFrame with pre-computed rolling returns.

    Returns:
        DataFrame with all raw features, or None on failure.
    """
    try:
        df = fetch_daily_bars(ticker, fetch_from, fetch_to)
    except Exception as exc:
        print(f"  [SKIP] {ticker}: load error â€” {exc}")
        return None

    if df.empty or len(df) < 80:
        print(f"  [SKIP] {ticker}: only {len(df)} daily bars")
        return None

    df = df.sort_values("Date").reset_index(drop=True)

    # --- Raw momentum factors ---
    df["Ticker"] = ticker
    df["Sector"] = sector
    df["Ret2"]   = df["Close"].pct_change(2)
    df["Ret3"]   = df["Close"].pct_change(3)
    df["Ret5"]   = df["Close"].pct_change(5)
    df["Ret10"]  = df["Close"].pct_change(10)
    df["AvgVol20D"] = df["Volume"].rolling(20).mean()
    df["RVOL"]   = df["Volume"] / df["AvgVol20D"].replace(0, np.nan)

    # Merge SPY returns
    df = df.merge(
        spy_returns[["Date", "SPY_Ret2", "SPY_Ret3", "SPY_Ret5", "SPY_Ret10"]],
        on="Date",
        how="inner",
    )

    df["RS2"]  = [safe_rs(a, b) for a, b in zip(df["Ret2"],  df["SPY_Ret2"])]
    df["RS3"]  = [safe_rs(a, b) for a, b in zip(df["Ret3"],  df["SPY_Ret3"])]
    df["RS5"]  = [safe_rs(a, b) for a, b in zip(df["Ret5"],  df["SPY_Ret5"])]
    df["RS10"] = [safe_rs(a, b) for a, b in zip(df["Ret10"], df["SPY_Ret10"])]

    # Liquidity filter (same as leader scanner)
    df = df[(df["Close"] >= 5) & (df["AvgVol20D"] >= 500_000)].copy()
    df = df.dropna(subset=["RS2", "RS3", "RS5", "RS10", "RVOL"]).copy()

    if df.empty:
        print(f"  [SKIP] {ticker}: no rows after liquidity filter")
        return None

    # --- Quality features (ATR, proximity, CLV, dollar vol, gap, SPY regime) ---
    # build_quality_features expects Date as index or column + Open/High/Low/Close/Volume
    spy_close_df = spy_returns[["Date", "Close"]].rename(columns={"Close": "Close"}).copy()
    spy_close_df = spy_close_df.set_index("Date")
    spy_close_df.index = pd.to_datetime(spy_close_df.index)

    # Lookup earnings dates for this ticker (None if not in calendar)
    earnings_dates = None
    if earnings_calendar:
        earnings_dates = earnings_calendar.get(ticker)

    quality_df = build_quality_features(
        stock_df=df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy(),
        spy_df=spy_close_df,
        earnings_dates=earnings_dates,
    )
    # quality_df is indexed by Date; merge back
    quality_cols = [c for c in quality_df.columns if c not in {"Open", "High", "Low", "Close", "Volume"}]
    quality_df = quality_df[quality_cols].reset_index().rename(columns={"index": "Date"})
    quality_df["Date"] = pd.to_datetime(quality_df["Date"])

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.merge(quality_df, on="Date", how="left")

    return df


# ---------------------------------------------------------------------------
# Cross-sectional percentile ranking
# ---------------------------------------------------------------------------

def add_percentile_ranks(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily cross-sectional percentile ranks (0â€“100) for momentum
    factors.  Applied after all tickers are concatenated.

    Adds columns: RS2_Pct, RS3_Pct, RS5_Pct, RS10_Pct, RVOL_Pct
    """
    rank_cols = ["RS2", "RS3", "RS5", "RS10", "RVOL"]
    pct_cols  = ["RS2_Pct", "RS3_Pct", "RS5_Pct", "RS10_Pct", "RVOL_Pct"]

    for raw, pct in zip(rank_cols, pct_cols):
        data[pct] = data.groupby("Date")[raw].rank(pct=True) * 100

    return data


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_research_dataset(
    universe_csv: str = str(ROOT / "data" / "minute_csv_1y_all1000" / "russell_1000.csv"),
    output_csv: str   = str(ROOT / "output" / "research_dataset.csv"),
    lookback_days: int = 365,
    warmup_days: int   = 90,
) -> pd.DataFrame:
    """
    Full pipeline: load universe â†’ fetch OHLCV â†’ compute all features â†’
    cross-sectional rank â†’ save CSV.

    Args:
        universe_csv:  Path to CSV with columns [ticker, sector].
        output_csv:    Destination CSV path.
        lookback_days: Number of trading days of data to keep in output.
        warmup_days:   Extra days to fetch before the cutoff so rolling
                       indicators (ATR20, SMA200, etc.) are properly seeded.

    Returns:
        Master research DataFrame.
    """
    universe_path = Path(universe_csv)
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    universe = pd.read_csv(universe_path)
    if not {"ticker", "sector"}.issubset(universe.columns):
        raise ValueError("Universe CSV must contain columns: ticker, sector")

    today = datetime.now().date()
    fetch_from = (today - timedelta(days=lookback_days + warmup_days)).strftime("%Y-%m-%d")
    fetch_to   = today.strftime("%Y-%m-%d")
    cutoff     = today - timedelta(days=lookback_days)

    print("=" * 60)
    print("BUILD RESEARCH DATASET â€” Leader Score V2")
    print("=" * 60)
    print(f"Universe : {universe_path} ({len(universe)} tickers)")
    print(f"Fetch    : {fetch_from} â†’ {fetch_to}")
    print(f"Cutoff   : {cutoff} (warmup excluded from output)")
    print(f"Output   : {output_csv}")

    # --- SPY ---
    print("\nFetching SPY...")
    spy_raw = fetch_daily_bars("SPY", fetch_from, fetch_to)
    if spy_raw.empty or len(spy_raw) < 80:
        raise ValueError("Not enough SPY data to proceed.")
    spy_raw = spy_raw.sort_values("Date").reset_index(drop=True)
    spy_returns = compute_spy_returns(spy_raw)
    print(f"SPY bars loaded: {len(spy_raw)} (latest: {spy_raw['Date'].iloc[-1]})")

    # --- Earnings calendar (optional â€” NaN placeholders if not yet fetched) ---
    earnings_calendar = load_earnings_calendar()
    if earnings_calendar:
        print(f"Earnings calendar loaded: {len(earnings_calendar)} tickers")
    else:
        print("Earnings calendar not found â€” earnings flags will be NaN.")
        print("Run: python leader_score_v2/features/fetch_earnings.py")

    # --- Per-ticker ---
    all_rows: list[pd.DataFrame] = []

    for i, row in universe.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        sector = str(row["sector"]).strip()

        if not ticker or ticker.lower() == "nan":
            continue

        print(f"[{i + 1:>4}/{len(universe)}] {ticker:<8}", end="  ")

        result = process_ticker(ticker, sector, fetch_from, fetch_to, spy_returns, earnings_calendar)

        if result is not None:
            all_rows.append(result)
            print(f"{len(result)} rows")
        # else: process_ticker already printed the skip reason

    if not all_rows:
        raise ValueError("No tickers produced valid data.")

    print(f"\nCombining {len(all_rows)} tickers...")
    data = pd.concat(all_rows, ignore_index=True)

    # Apply cutoff (drop warmup period)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[data["Date"].dt.date >= cutoff].copy()
    print(f"Rows after cutoff: {len(data)}")

    # Cross-sectional percentile ranks
    print("Computing cross-sectional percentile ranks...")
    data = add_percentile_ranks(data)

    # Final column ordering
    id_cols      = ["Date", "Ticker", "Sector"]
    ohlcv_cols   = ["Open", "High", "Low", "Close", "Volume"]
    momentum_raw = ["AvgVol20D", "RVOL", "RS2", "RS3", "RS5", "RS10"]
    momentum_pct = ["RS2_Pct", "RS3_Pct", "RS5_Pct", "RS10_Pct", "RVOL_Pct"]
    quality_cols = [
        "ATR20", "Today_Range", "ATR_Expansion",
        "High52", "High52_Proximity",
        "AvgRange5", "AvgRange20", "VolatilityContraction",
        "Daily_CLV",
        "DollarVolume", "DollarVolume20",
        "GapPct", "GapExtensionATR",
        "SPY_Close", "SPY_SMA50", "SPY_SMA200",
        "SPY_Bull", "SPY_Bear", "SPY_Neutral",
        "DaysSinceEarnings", "DaysUntilNextEarnings",
        "RecentEarningsFlag", "UpcomingEarningsFlag",
    ]

    keep = id_cols + ohlcv_cols + momentum_raw + momentum_pct + quality_cols
    # Only keep columns that actually exist (guards against optional cols)
    keep = [c for c in keep if c in data.columns]
    data = data[keep].sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Save
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)

    print(f"\nResearch dataset saved â†’ {out_path}")
    print(f"Shape  : {data.shape}")
    print(f"Dates  : {data['Date'].min().date()} â†’ {data['Date'].max().date()}")
    print(f"Tickers: {data['Ticker'].nunique()}")
    print(f"Columns: {list(data.columns)}")

    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Leader Score V2 research dataset")
    parser.add_argument(
        "--universe",
        default=str(ROOT / "data" / "minute_csv_1y_all1000" / "russell_1000.csv"),
        help="Path to universe CSV with ticker,sector columns",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "output" / "research_dataset.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=365,
        help="Days of history to keep in output (default 365)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=90,
        help="Extra warmup days to seed rolling indicators (default 90)",
    )
    args = parser.parse_args()

    build_research_dataset(
        universe_csv=args.universe,
        output_csv=args.output,
        lookback_days=args.lookback,
        warmup_days=args.warmup,
    )

