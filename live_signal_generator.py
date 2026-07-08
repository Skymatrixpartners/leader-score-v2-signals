"""
Live Signal Generator — Leader Score V2
========================================
Runs at ~3:25 PM ET daily. Fetches the last 60 days of daily bars
for all Russell 1000 tickers, recomputes all features + Leader_Score_V2,
and outputs today's BUY signals to output/live_signals_YYYY-MM-DD.csv.

No local data files required — all data is fetched via the Massive API.

Usage
-----
    python live_signal_generator.py
    python live_signal_generator.py --date 2026-07-07
    python live_signal_generator.py --output my_signals.csv
    python live_signal_generator.py --min-score 85 --top-n 5

Environment
-----------
    MASSIVE_API_KEY   required
    MASSIVE_API_URL   optional (default: https://api.massive.com/v2)

Cloud deployment: see .github/workflows/daily_signals.yml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# .env can be in trail_live/ or one level up at project/.env
for env_candidate in [ROOT / ".env", ROOT.parent / "project" / ".env"]:
    if env_candidate.exists():
        load_dotenv(env_candidate)
        break

API_KEY  = os.getenv("MASSIVE_API_KEY", "")
BASE_URL = os.getenv("MASSIVE_API_URL", "https://api.massive.com/v2").rstrip("/")

UNIVERSE_FILE  = ROOT / "data" / "russell_1000.csv"
OUTPUT_DIR     = ROOT / "output" / "signals"
LOOKBACK_DAYS  = 90    # fetch 90 calendar days -> ~63 trading days -> covers RS10+ATR20+warmup
SPY_LOOKBACK   = 310   # fetch 310 calendar days for SPY -> covers SMA50 + SMA200
RATE_LIMIT     = 0.12  # seconds between API calls
MAX_WORKERS    = 8

# Score bands (sweep-validated)
BAND_D_LO, BAND_D_HI = 80.0,  84.0   # 1x  — momentum base zone (no leverage)
BAND_A_LO, BAND_A_HI = 90.0,  95.0   # 2x  — high-conviction zone
BAND_B_LO, BAND_B_HI = 98.0, 100.0   # 1x  — moonshots (no leverage)
# Regime minimum scores
REGIME_BULL_MIN       = 80.0
REGIME_NEUTRAL_MIN    = 85.0

# Score weights (IC-validated, must sum to 1.0)
WEIGHTS = {
    "RS10_Pct":         0.45,
    "RS5_Pct":          0.25,
    "RVOL_Pct":         0.20,
    "RS3_Pct":          0.05,
    "DollarVolume_Pct": 0.05,
}

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _request(url: str, params: dict, retries: int = 4) -> dict:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            sleep = min(2 ** attempt, 30)
            print(f"  API retry {attempt}/{retries}: {exc}", flush=True)
            time.sleep(sleep)
    return {}


def fetch_daily_bars(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLCV from Massive API. Returns empty DataFrame on failure."""
    url = f"{BASE_URL}/aggs/ticker/{ticker}/range/1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": API_KEY}
    rows, next_url = [], url
    while next_url:
        payload = _request(next_url, params)
        rows.extend(payload.get("results", []))
        next_url = payload.get("next_url")
        if next_url:
            params = {"apiKey": API_KEY}
            time.sleep(RATE_LIMIT)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Date"]   = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
    return df[["Date","Open","High","Low","Close","Volume"]].sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature computation (self-contained, no imports from features/)
# ---------------------------------------------------------------------------

def compute_ticker_features(
    ticker: str,
    sector: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute all momentum + quality features for one ticker.
    df and spy_df are daily OHLCV DataFrames sorted ascending by Date.
    Returns rows for the LAST date only (today's signal row).
    """
    if df.empty or len(df) < 12:
        return pd.DataFrame()

    df = df.copy().sort_values("Date").reset_index(drop=True)
    spy = spy_df.copy().sort_values("Date").reset_index(drop=True)

    # Align on common dates
    merged = df.merge(spy[["Date","Close"]].rename(columns={"Close":"SPY_Close"}),
                      on="Date", how="inner")
    if len(merged) < 12:
        return pd.DataFrame()

    merged = merged.sort_values("Date").reset_index(drop=True)

    # Ticker returns
    merged["Ret2"]  = merged["Close"].pct_change(2)
    merged["Ret3"]  = merged["Close"].pct_change(3)
    merged["Ret5"]  = merged["Close"].pct_change(5)
    merged["Ret10"] = merged["Close"].pct_change(10)

    # SPY returns
    merged["SPY_Ret2"]  = merged["SPY_Close"].pct_change(2)
    merged["SPY_Ret3"]  = merged["SPY_Close"].pct_change(3)
    merged["SPY_Ret5"]  = merged["SPY_Close"].pct_change(5)
    merged["SPY_Ret10"] = merged["SPY_Close"].pct_change(10)

    # Relative strength vs SPY
    merged["RS2"]  = merged["Ret2"]  - merged["SPY_Ret2"]
    merged["RS3"]  = merged["Ret3"]  - merged["SPY_Ret3"]
    merged["RS5"]  = merged["Ret5"]  - merged["SPY_Ret5"]
    merged["RS10"] = merged["Ret10"] - merged["SPY_Ret10"]

    # Volume metrics
    merged["AvgVol20D"] = merged["Volume"].rolling(20, min_periods=10).mean()
    merged["RVOL"]      = merged["Volume"] / merged["AvgVol20D"].replace(0, np.nan)
    merged["DollarVolume"] = merged["Close"] * merged["Volume"]

    # ATR20
    tr = pd.concat([
        merged["High"] - merged["Low"],
        (merged["High"] - merged["Close"].shift(1)).abs(),
        (merged["Low"]  - merged["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    merged["ATR20"] = tr.rolling(20, min_periods=10).mean()

    # Keep only the last row (today)
    last = merged.tail(1).copy()
    last["Ticker"] = ticker
    last["Sector"] = sector

    return last[[
        "Date","Ticker","Sector","Close","Volume",
        "RS2","RS3","RS5","RS10",
        "RVOL","AvgVol20D","DollarVolume","ATR20",
        "SPY_Close",
    ]]


def build_spy_regime(spy_df: pd.DataFrame, as_of: date) -> dict:
    """Compute SPY SMA50, SMA200 and regime classification for a given date."""
    spy = spy_df[spy_df["Date"] <= pd.Timestamp(as_of)].tail(210).copy()
    if len(spy) < 50:
        return {"regime": "BEAR", "spy_close": None, "sma50": None, "sma200": None}
    close      = spy["Close"].iloc[-1]
    sma50      = spy["Close"].tail(50).mean()
    sma200     = spy["Close"].tail(200).mean() if len(spy) >= 200 else None
    spy_bull   = (sma200 is not None) and (close > sma50) and (sma50 > sma200)
    spy_bear   = (sma200 is not None) and (close < sma200)
    if spy_bull:
        regime = "BULL"
    elif spy_bear:
        regime = "BEAR"
    else:
        regime = "NEUTRAL"
    return {
        "regime": regime,
        "spy_close": round(close, 2),
        "sma50":     round(sma50, 2),
        "sma200":    round(sma200, 2) if sma200 else None,
        "spy_bull":  spy_bull,
        "spy_bear":  spy_bear,
    }


# ---------------------------------------------------------------------------
# Cross-sectional percentile ranks
# ---------------------------------------------------------------------------

def add_pct_ranks(df: pd.DataFrame) -> pd.DataFrame:
    for col, ascending in [
        ("RS10",        True),
        ("RS5",         True),
        ("RS3",         True),
        ("RVOL",        True),
        ("DollarVolume",True),
    ]:
        if col in df.columns:
            df[f"{col}_Pct"] = df[col].rank(pct=True, ascending=ascending) * 100
    return df


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def compute_score(df: pd.DataFrame) -> pd.Series:
    score = sum(df[col] * w for col, w in WEIGHTS.items() if col in df.columns)
    return score


# ---------------------------------------------------------------------------
# Signal filter
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, regime: str, top_n: int) -> pd.DataFrame:
    """Apply regime gate + band filter + top-N selection."""
    if regime == "BEAR":
        print("  REGIME: BEAR — no trades today.")
        return pd.DataFrame()

    min_score = REGIME_BULL_MIN if regime == "BULL" else REGIME_NEUTRAL_MIN

    # Band D: score 80-84 → 1x (no leverage)
    band_d = df[
        (df["Leader_Score_V2"] >= max(BAND_D_LO, min_score)) &
        (df["Leader_Score_V2"] <= BAND_D_HI)
    ].copy()
    band_d["Band"]     = "D"
    band_d["Leverage"] = 1.0

    # Band A: score 90-95 → 2x leverage (high-conviction zone)
    band_a = df[
        (df["Leader_Score_V2"] >= max(BAND_A_LO, min_score)) &
        (df["Leader_Score_V2"] <= BAND_A_HI)
    ].copy()
    band_a["Band"]     = "A"
    band_a["Leverage"] = 2.0

    # Band B: score 98-100 → 1x (no leverage, moonshots)
    band_b = df[
        (df["Leader_Score_V2"] >= max(BAND_B_LO, min_score)) &
        (df["Leader_Score_V2"] <= BAND_B_HI)
    ].copy()
    band_b["Band"]     = "B"
    band_b["Leverage"] = 1.0

    combined = pd.concat([band_d, band_a, band_b], ignore_index=True)
    if combined.empty:
        print("  No stocks pass band filter today.")
        return pd.DataFrame()

    # Top-N by score across all bands
    combined = combined.sort_values("Leader_Score_V2", ascending=False).head(top_n)
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leader Score V2 — Live Signal Generator")
    p.add_argument("--date",      default=None,
                   help="Signal date YYYY-MM-DD (default: today)")
    p.add_argument("--output",    default=None,
                   help="Output CSV path (default: output/signals/live_signals_DATE.csv)")
    p.add_argument("--universe",  default=str(UNIVERSE_FILE),
                   help="Universe CSV with ticker,sector columns")
    p.add_argument("--lookback",  type=int, default=LOOKBACK_DAYS,
                   help="Calendar days of history to fetch (default: 60)")
    p.add_argument("--workers",   type=int, default=8,
                   help="Parallel API fetch workers (default: 8)")
    p.add_argument("--top-n",     type=int, default=10,
                   help="Max signals to emit (default: 10)")
    p.add_argument("--min-score", type=float, default=None,
                   help="Override minimum score threshold")
    p.add_argument("--no-regime-gate", action="store_true",
                   help="Ignore SPY regime, score all days")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not API_KEY:
        print("ERROR: MASSIVE_API_KEY not set. Add to .env or set environment variable.")
        sys.exit(1)

    signal_date = (
        date.fromisoformat(args.date) if args.date else date.today()
    )
    fetch_start = signal_date - timedelta(days=args.lookback)  # for tickers
    fetch_end   = signal_date

    print("=" * 60)
    print(f"  LEADER SCORE V2 — LIVE SIGNALS")
    print(f"  Signal date : {signal_date}")
    print(f"  Fetch range : {fetch_start} to {fetch_end}")
    print("=" * 60)

    # Load universe
    universe_path = Path(args.universe)
    if not universe_path.exists():
        # Fallback: look for russell_1000.csv in common locations
        for candidate in [
            ROOT / "data" / "russell_1000.csv",
            ROOT.parents[1] / "leader_score_v2" / "data" / "minute_csv_1y_all1000" / "russell_1000.csv",
        ]:
            if candidate.exists():
                universe_path = candidate
                break
        else:
            print(f"ERROR: Universe file not found at {args.universe}")
            sys.exit(1)

    universe = pd.read_csv(universe_path)
    tickers  = universe["ticker"].dropna().str.upper().unique().tolist()
    sectors  = dict(zip(universe["ticker"].str.upper(), universe.get("sector", "")))
    print(f"  Universe    : {len(tickers)} tickers from {universe_path.name}")

    # Fetch SPY first (needed for RS calculation)
    print(f"\nFetching SPY ({fetch_start} to {fetch_end}) ...")
    spy_df = fetch_daily_bars("SPY", fetch_start, fetch_end)
    if spy_df.empty or len(spy_df) < 10:
        print("ERROR: Could not fetch SPY data.")
        sys.exit(1)
    print(f"  SPY rows: {len(spy_df)}")

    # Regime (uses full 310-day SPY for SMA200)
    spy_fetch_start = signal_date - timedelta(days=SPY_LOOKBACK)
    if spy_fetch_start < fetch_start:
        print(f"Fetching extended SPY history ({spy_fetch_start} to {fetch_end}) for SMA200 ...")
        spy_df_regime = fetch_daily_bars("SPY", spy_fetch_start, fetch_end)
        if spy_df_regime.empty:
            spy_df_regime = spy_df
    else:
        spy_df_regime = spy_df
    regime_info = build_spy_regime(spy_df_regime, signal_date)
    regime = regime_info["regime"] if not args.no_regime_gate else "BULL"
    print(f"\n  SPY Close : {regime_info['spy_close']}")
    print(f"  SMA50     : {regime_info['sma50']}")
    print(f"  SMA200    : {regime_info['sma200']}")
    print(f"  Regime    : {regime}")

    if regime == "BEAR" and not args.no_regime_gate:
        print("\n  REGIME: BEAR — emitting empty signal file.")
        signals = pd.DataFrame(columns=["Ticker","Band","Leader_Score_V2","Leverage","Close","ATR20","Sector"])
        _save_and_print(signals, signal_date, regime_info, args)
        return

    # Fetch all tickers in parallel
    print(f"\nFetching {len(tickers)} tickers ({args.workers} workers) ...")
    all_rows: list[pd.DataFrame] = []
    failed = 0
    _lock = __import__("threading").Lock()

    def _fetch_one(ticker: str) -> pd.DataFrame | None:
        bars = fetch_daily_bars(ticker, fetch_start, fetch_end)
        time.sleep(RATE_LIMIT)
        if bars.empty:
            return None
        return compute_ticker_features(ticker, sectors.get(ticker, ""), bars, spy_df)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            completed += 1
            try:
                row = fut.result()
                if row is not None and not row.empty:
                    with _lock:
                        all_rows.append(row)
                else:
                    with _lock:
                        failed += 1
            except Exception:
                with _lock:
                    failed += 1
            if completed % 100 == 0:
                print(f"  {completed}/{len(tickers)} fetched  ({failed} failed)", flush=True)

    print(f"  Done. {len(all_rows)} tickers with valid data, {failed} failed.")

    if not all_rows:
        print("ERROR: No ticker data retrieved.")
        sys.exit(1)

    # Build universe DataFrame for today
    df = pd.concat(all_rows, ignore_index=True)
    # Keep only rows matching signal_date (last trading day <= signal_date)
    max_date = df["Date"].max()
    df = df[df["Date"] == max_date].copy()
    print(f"  Using data as of: {max_date.date()}")

    if df.empty:
        print("ERROR: No rows for today's date.")
        sys.exit(1)

    # Cross-sectional percentile ranks
    df = add_pct_ranks(df)

    # Leader Score
    df["Leader_Score_V2"] = compute_score(df)

    # Apply filters
    df_signals = apply_filters(df, regime, args.top_n)

    _save_and_print(df_signals, signal_date, regime_info, args)


def _save_and_print(
    signals: pd.DataFrame,
    signal_date: date,
    regime_info: dict,
    args: argparse.Namespace,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.output)
        if args.output
        else OUTPUT_DIR / f"live_signals_{signal_date}.csv"
    )

    # Build output table
    if not signals.empty:
        out = signals[[
            "Ticker", "Band", "Leader_Score_V2", "Leverage",
            "RS10_Pct", "RS5_Pct", "RVOL_Pct", "RS3_Pct", "DollarVolume_Pct",
            "Close", "ATR20", "Sector",
        ]].copy()
        out["Stop_Price"]    = (out["Close"] - 2.0 * out["ATR20"]).round(2)
        out["Signal_Date"]   = str(signal_date)
        out["Regime"]        = regime_info["regime"]
        out["SPY_Close"]     = regime_info["spy_close"]
        out["SPY_SMA50"]     = regime_info["sma50"]
        out["SPY_SMA200"]    = regime_info["sma200"]
        out["Leader_Score_V2"] = out["Leader_Score_V2"].round(2)
        for col in ["RS10_Pct","RS5_Pct","RVOL_Pct","RS3_Pct","DollarVolume_Pct"]:
            if col in out.columns:
                out[col] = out[col].round(1)
        out.to_csv(out_path, index=False)
    else:
        pd.DataFrame(columns=["Ticker","Band","Leader_Score_V2","Leverage",
                               "Signal_Date","Regime","SPY_Close"]).to_csv(out_path, index=False)

    # Print summary
    print()
    print("=" * 60)
    print(f"  SIGNALS — {signal_date}  |  Regime: {regime_info['regime']}")
    print("=" * 60)
    if signals.empty:
        print("  No signals today.")
    else:
        print(f"  {'Ticker':<8} {'Band':<6} {'Score':>6} {'Lev':>5} {'Close':>8} {'Stop':>8} {'Sector'}")
        print(f"  {'-'*7:<8} {'-'*5:<6} {'-'*5:>6} {'-'*4:>5} {'-'*7:>8} {'-'*7:>8} {'-'*10}")
        for _, r in signals.iterrows():
            stop = round(r["Close"] - 2.0 * r["ATR20"], 2) if "ATR20" in r and pd.notna(r["ATR20"]) else "N/A"
            print(f"  {r['Ticker']:<8} {r['Band']:<6} {r['Leader_Score_V2']:>6.2f} "
                  f"{r['Leverage']:>5.1f}x {r['Close']:>8.2f} {str(stop):>8} {r.get('Sector','')}")
    print()
    print(f"  Saved -> {out_path}")
    print()

    # Also write a latest.json for webhook/dashboard consumption
    latest = {
        "date":    str(signal_date),
        "regime":  regime_info["regime"],
        "spy":     regime_info,
        "signals": signals[["Ticker","Band","Leader_Score_V2","Leverage","Close","ATR20","Sector"]].to_dict(orient="records") if not signals.empty else [],
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(latest, indent=2, default=str))
    print(f"  JSON   -> {json_path}")


if __name__ == "__main__":
    main()
