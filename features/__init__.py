"""
fetch_russell_ohlc.py
---------------------
Step 1: Fetches top 1000 US stocks by market cap (Russell-style universe)
Step 2: Fetches 1-min OHLC for last 6 months for each stock (50 parallel workers)
Step 3: Saves each ticker as its own .parquet file

- Regular market hours only: 9:30 AM - 3:59 PM EST
- Columns: Date, Time, Open, High, Low, Close
- Resume-safe: skips tickers already downloaded
- Auto-paging: handles any number of pages per ticker
- Parallel: 50 tickers fetched simultaneously

Requirements:
    pip install requests pandas pytz pyarrow
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
API_KEY      = "lYkOOYWztfYCJPLEesOHfn_jR3XAxG9K"   # <-- replace with your new key
BASE_URL     = "https://api.massive.com"
MAX_WORKERS  = 50                                      # parallel tickers at once

RUSSELL_FILE = Path("data/russell_1000.csv")
CACHE_FILE   = Path("data/market_cap_cache.csv")
OUTPUT_DIR   = Path("parquet_data")

TARGET_COUNT = 1000

# ─────────────────────────────────────────────
# DATE RANGE — last 6 months
# ─────────────────────────────────────────────
END_DATE   = datetime.today()
START_DATE = END_DATE - timedelta(days=182)

start_str = START_DATE.strftime("%Y-%m-%d")
end_str   = END_DATE.strftime("%Y-%m-%d")

# Thread-safe print lock (prevents garbled output from parallel threads)
print_lock = threading.Lock()

def tprint(msg):
    with print_lock:
        print(msg)

# ─────────────────────────────────────────────
# SAFE REQUEST (retry on failure / rate limit)
# ─────────────────────────────────────────────
def safe_get(url, params=None, retries=5, sleep_seconds=3):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = sleep_seconds * attempt
                tprint(f"    Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            tprint(f"    Request failed attempt {attempt}/{retries}: {e}")
            time.sleep(sleep_seconds * attempt)
    return None

# ─────────────────────────────────────────────
# STEP 1A — GET ALL ACTIVE US STOCKS
# ─────────────────────────────────────────────
def get_all_active_us_stocks():
    url    = f"{BASE_URL}/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "type":   "CS",
        "limit":  1000,
        "apiKey": API_KEY,
    }
    all_rows = []

    while True:
        r = safe_get(url, params=params)
        if r is None:
            break
        data = r.json()
        all_rows.extend(data.get("results", []))
        next_url = data.get("next_url")
        if not next_url:
            break
        url    = next_url
        params = {"apiKey": API_KEY}
        time.sleep(0.5)

    return all_rows

# ─────────────────────────────────────────────
# STEP 1B — GET TICKER DETAILS (market cap)
# ─────────────────────────────────────────────
def get_ticker_details(ticker):
    url = f"{BASE_URL}/v3/reference/tickers/{ticker}"
    r   = safe_get(url, params={"apiKey": API_KEY})
    if r is None:
        return None
    return r.json().get("results", {})

# ─────────────────────────────────────────────
# STEP 1 — BUILD RUSSELL-STYLE 1000 UNIVERSE
# ─────────────────────────────────────────────
def load_cache():
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame(columns=["ticker", "sector", "market_cap"])

def save_cache(df):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)

def build_russell_universe():
    if RUSSELL_FILE.exists():
        df = pd.read_csv(RUSSELL_FILE)
        print(f"  Loaded existing universe: {len(df)} tickers from {RUSSELL_FILE}")
        return df["ticker"].tolist()

    print("  Building Russell-style 1000 universe...")
    cache = load_cache()

    if len(cache) >= TARGET_COUNT:
        print("  Using existing cache.")
        df = cache.copy()
    else:
        stocks = get_all_active_us_stocks()
        print(f"  Total active common stocks found: {len(stocks)}")

        existing_tickers = set(cache["ticker"].astype(str).str.upper())
        rows             = cache.to_dict("records")

        for i, stock in enumerate(stocks, start=1):
            ticker = str(stock.get("ticker", "")).strip().upper()
            if not ticker or ticker in existing_tickers:
                continue

            print(f"  [{i}/{len(stocks)}] Fetching details: {ticker}")
            details = get_ticker_details(ticker)
            if not details:
                continue

            market_cap = details.get("market_cap")
            sector     = details.get("sic_description", "Unknown") or "Unknown"

            if market_cap is None:
                continue

            rows.append({"ticker": ticker, "sector": sector, "market_cap": market_cap})
            existing_tickers.add(ticker)

            cache_df    = pd.DataFrame(rows)
            save_cache(cache_df)
            valid_count = cache_df["market_cap"].notna().sum()
            print(f"    Valid stocks so far: {valid_count}")

            if valid_count >= TARGET_COUNT:
                break

            time.sleep(0.5)

        df = pd.DataFrame(rows)

    df       = df.dropna(subset=["market_cap"])
    df       = df.sort_values("market_cap", ascending=False).head(TARGET_COUNT)
    universe = df[["ticker", "sector"]].copy()

    RUSSELL_FILE.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(RUSSELL_FILE, index=False)
    print(f"  Saved {len(universe)} tickers to {RUSSELL_FILE}")

    return universe["ticker"].tolist()

# ─────────────────────────────────────────────
# STEP 2 — FETCH 1-MIN OHLC FOR ONE TICKER
# ─────────────────────────────────────────────
def fetch_bars(ticker):
    url    = (
        f"{BASE_URL}/v2/aggs/ticker/{ticker}"
        f"/range/1/minute/{start_str}/{end_str}"
    )
    params      = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}
    all_results = []
    page        = 1

    while url:
        r = safe_get(url, params=params)
        if r is None:
            tprint(f"    [ERROR] Failed to fetch page {page} for {ticker}")
            break

        data    = r.json()
        results = data.get("results", [])
        all_results.extend(results)
        tprint(f"  {ticker} | Page {page} — {len(results):>6} bars (total: {len(all_results):>7})")

        url    = data.get("next_url")
        params = {"apiKey": API_KEY}
        page  += 1
        if url:
            time.sleep(0.25)

    return all_results

# ─────────────────────────────────────────────
# STEP 3 — PROCESS & SAVE AS PARQUET
# ─────────────────────────────────────────────
def process_and_save(ticker, bars):
    if not bars:
        return False

    df = pd.DataFrame(bars)

    df["datetime"] = (
        pd.to_datetime(df["t"], unit="ms", utc=True)
        .dt.tz_convert("US/Eastern")
    )

    # Regular market hours only: 9:30 AM - 3:59 PM
    df = df[
        (df["datetime"].dt.time >= pd.Timestamp("09:30").time()) &
        (df["datetime"].dt.time <= pd.Timestamp("15:59").time())
    ].copy()

    if df.empty:
        return False

    df["Date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    df["Time"] = df["datetime"].dt.strftime("%H:%M")

    df = df[["Date", "Time", "o", "h", "l", "c"]].copy()
    df.columns = ["Date", "Time", "Open", "High", "Low", "Close"]
    df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]].round(2)
    df = df.reset_index(drop=True)

    filename = f"{ticker}_1min_OHLC_{start_str}_to_{end_str}.parquet"
    filepath = OUTPUT_DIR / filename
    df.to_parquet(filepath, index=False)

    tprint(f"  ✓ {ticker} — {len(df):,} rows saved")
    return True

# ─────────────────────────────────────────────
# COMBINED FETCH + SAVE (one function per thread)
# ─────────────────────────────────────────────
def fetch_and_save(ticker):
    try:
        bars = fetch_bars(ticker)
        ok   = process_and_save(ticker, bars)
        if not ok:
            tprint(f"  [SKIP] {ticker} — no usable data")
        return ticker, ok
    except Exception as e:
        tprint(f"  [ERROR] {ticker} — {e}")
        return ticker, False

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build universe ────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: Building Russell-style 1000 universe")
    print("=" * 60)
    tickers = build_russell_universe()
    print(f"\n  Total tickers to process: {len(tickers)}")

    # ── Resume: skip already downloaded ──────
    already_done = {
        f.stem.split("_")[0]
        for f in OUTPUT_DIR.glob("*.parquet")
    }
    remaining = [t for t in tickers if t not in already_done]

    print(f"  Already downloaded : {len(already_done)}")
    print(f"  Remaining          : {len(remaining)}")

    if not remaining:
        print("\n  All tickers already downloaded. Nothing to do.")
        return

    # ── Step 2 & 3: Parallel fetch + save ────
    print("\n" + "=" * 60)
    print("  STEP 2: Fetching 1-min OHLC data")
    print(f"  Date range   : {start_str}  ->  {end_str}")
    print(f"  Workers      : {MAX_WORKERS} parallel tickers")
    print(f"  Output dir   : {OUTPUT_DIR.resolve()}")
    print("=" * 60 + "\n")

    success = 0
    failed  = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_and_save, ticker): ticker for ticker in remaining}

        for future in as_completed(futures):
            ticker, ok = future.result()
            if ok:
                success += 1
            else:
                failed.append(ticker)

            total_done = success + len(failed)
            tprint(f"  Progress: {total_done}/{len(remaining)} done "
                   f"({success} success, {len(failed)} failed)")

    # ── Final Summary ─────────────────────────
    print("\n" + "=" * 60)
    print(f"  ALL DONE!")
    print(f"  Successful : {success + len(already_done)} tickers")
    print(f"  Failed     : {len(failed)} tickers")
    if failed:
        print(f"  Failed list: {', '.join(failed)}")
    print(f"\n  Parquet files saved in:")
    print(f"  {OUTPUT_DIR.resolve()}")
    print("=" * 60)
    print("\n  To read any file later:")
    print("  import pandas as pd")
    print(f"  df = pd.read_parquet('parquet_data/AAPL_1min_OHLC_{start_str}_to_{end_str}.parquet')")

if __name__ == "__main__":
    main()