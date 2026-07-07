"""
Fetch Earnings Calendar from Massive API (Polygon-compatible)

Downloads historical quarterly earnings dates for all tickers in the
universe and saves a master earnings_calendar.csv.

Data source
-----------
Endpoint : GET /vX/reference/financials
Fields   : filing_date, period_of_report_date, eps_actual
Note     : filing_date = SEC 10-Q/10-K filing date, typically 2â€“5 days
           AFTER the actual earnings press release.  For 5-day proximity
           flags this is accurate enough for backtesting.

Output
------
leader_score_v2/data/earnings_calendar.csv
Columns: ticker, earnings_date, period_end_date, eps_actual

Usage
-----
    python leader_score_v2/features/fetch_earnings.py

    # or with a custom universe:
    python leader_score_v2/features/fetch_earnings.py \\
        --universe leader_score_v2/data/minute_csv_1y_all1000/russell_1000.csv \\
        --output   leader_score_v2/data/earnings_calendar.csv \\
        --lookback 548
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY  = os.getenv("MASSIVE_API_KEY") or "lYkOOYWztfYCJPLEesOHfn_jR3XAxG9K"
BASE_URL = os.getenv("MASSIVE_API_URL", "https://api.massive.com")

if not API_KEY:
    raise EnvironmentError("MASSIVE_API_KEY not set. Add it to your .env file.")

DEFAULT_UNIVERSE = str(ROOT / "data" / "minute_csv_1y_all1000" / "russell_1000.csv")
DEFAULT_OUTPUT   = str(ROOT / "data" / "earnings_calendar.csv")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def safe_get(url: str, params: dict, retries: int = 5) -> dict | None:
    """
    GET request with retry + rate-limit handling.

    Returns parsed JSON dict, or None on persistent failure.
    """
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = 3 * attempt
                print(f"    Rate limited â€” waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            print(f"    Request error attempt {attempt}/{retries}: {exc}")
            time.sleep(3 * attempt)
    return None


def fetch_earnings_for_ticker(
    ticker: str,
    from_date: str,
    to_date: str,
) -> list[dict]:
    """
    Fetch quarterly financials for one ticker using Polygon's
    /vX/reference/financials endpoint.

    Extracts:
        - filing_date       : SEC filing date (proxy for earnings date)
        - period_end_date   : End of fiscal quarter
        - eps_actual        : Basic EPS reported (or NaN if unavailable)

    Args:
        ticker:    Ticker symbol (e.g. "AAPL").
        from_date: Start date YYYY-MM-DD.
        to_date:   End date YYYY-MM-DD.

    Returns:
        List of dicts with keys: ticker, earnings_date, period_end_date, eps_actual
    """
    url = f"{BASE_URL}/vX/reference/financials"
    params = {
        "ticker":              ticker,
        "timeframe":           "quarterly",
        "filing_date.gte":     from_date,
        "filing_date.lte":     to_date,
        "limit":               20,       # max 4 quarters/yr Ã— ~1.5yr = 6 needed
        "sort":                "filing_date",
        "order":               "desc",
        "apiKey":              API_KEY,
    }

    rows = []

    while True:
        data = safe_get(url, params)
        if data is None:
            break

        for result in data.get("results", []):
            filing_date    = result.get("filing_date")
            period_end     = result.get("period_of_report_date")

            # Extract basic EPS from nested financials object
            eps_actual = None
            try:
                eps_actual = (
                    result["financials"]
                    ["income_statement"]
                    ["basic_earnings_per_share"]
                    ["value"]
                )
            except (KeyError, TypeError):
                pass

            if filing_date:
                rows.append({
                    "ticker":         ticker,
                    "earnings_date":  filing_date,
                    "period_end_date": period_end,
                    "eps_actual":     eps_actual,
                })

        # Polygon paginates via next_url
        next_url = data.get("next_url")
        if not next_url:
            break
        url    = next_url
        params = {"apiKey": API_KEY}
        time.sleep(0.2)

    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def fetch_all_earnings(
    universe_csv: str = DEFAULT_UNIVERSE,
    output_csv: str   = DEFAULT_OUTPUT,
    lookback_days: int = 548,   # ~18 months so we have warm-up history
    sleep_seconds: float = 0.15,
) -> pd.DataFrame:
    """
    Fetch earnings calendar for all tickers in the universe and save CSV.

    The output CSV is used by features/events.py to compute
    DaysSinceEarnings, DaysUntilNextEarnings, RecentEarningsFlag,
    and UpcomingEarningsFlag for each stock on each trading day.

    Args:
        universe_csv:  Path to CSV with [ticker, sector] columns.
        output_csv:    Destination CSV path.
        lookback_days: How far back to fetch earnings (default 548 = ~18 months).
        sleep_seconds: Pause between API calls to respect rate limits.

    Returns:
        DataFrame with columns: ticker, earnings_date, period_end_date, eps_actual
    """
    universe_path = Path(universe_csv)
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    universe = pd.read_csv(universe_path)
    tickers  = universe["ticker"].dropna().str.strip().str.upper().tolist()

    today     = datetime.now().date()
    from_date = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_date   = today.strftime("%Y-%m-%d")

    out_path  = Path(output_csv)

    # Resume-safe: load already-fetched tickers if output file exists
    done_tickers: set[str] = set()
    existing_rows: list[dict] = []
    if out_path.exists():
        existing_df  = pd.read_csv(out_path)
        done_tickers = set(existing_df["ticker"].str.upper().unique())
        existing_rows = existing_df.to_dict("records")
        print(f"Resuming â€” {len(done_tickers)} tickers already fetched.")

    print("=" * 60)
    print("FETCH EARNINGS CALENDAR â€” Massive API")
    print("=" * 60)
    print(f"Universe   : {universe_path} ({len(tickers)} tickers)")
    print(f"Date range : {from_date} â†’ {to_date}")
    print(f"Output     : {out_path}")
    print()

    all_rows = list(existing_rows)

    for i, ticker in enumerate(tickers, start=1):
        if ticker in done_tickers:
            print(f"[{i:>4}/{len(tickers)}] {ticker:<8}  (already fetched â€” skip)")
            continue

        print(f"[{i:>4}/{len(tickers)}] {ticker:<8}", end="  ")

        rows = fetch_earnings_for_ticker(ticker, from_date, to_date)
        all_rows.extend(rows)

        print(f"{len(rows)} records")

        # Save incrementally after each ticker
        if all_rows:
            pd.DataFrame(all_rows).to_csv(out_path, index=False)

        time.sleep(sleep_seconds)

    if not all_rows:
        print("No earnings data found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["earnings_date"]   = pd.to_datetime(df["earnings_date"])
    df["period_end_date"] = pd.to_datetime(df["period_end_date"])
    df = df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print()
    print(f"Earnings calendar saved â†’ {out_path}")
    print(f"Total records : {len(df)}")
    print(f"Tickers       : {df['ticker'].nunique()}")
    print(f"Date range    : {df['earnings_date'].min().date()} â†’ {df['earnings_date'].max().date()}")
    print()
    print(df.head(10).to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch earnings calendar from Massive API")
    parser.add_argument("--universe",  default=DEFAULT_UNIVERSE, help="Universe CSV path")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT,   help="Output CSV path")
    parser.add_argument("--lookback",  type=int, default=548,    help="Days to look back (default 548 = ~18 months)")
    parser.add_argument("--sleep",     type=float, default=0.15, help="Sleep between API calls (default 0.15s)")
    args = parser.parse_args()

    fetch_all_earnings(
        universe_csv=args.universe,
        output_csv=args.output,
        lookback_days=args.lookback,
        sleep_seconds=args.sleep,
    )

