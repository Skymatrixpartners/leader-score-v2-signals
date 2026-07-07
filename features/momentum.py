import os
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv


# Load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("MASSIVE_API_KEY")
BASE_URL = os.getenv("MASSIVE_API_URL", "https://api.massive.com/v2")
PROJECT_DIR = Path(__file__).resolve().parents[2]
# Minute CSVs live in leader_score_v2/data/minute_csv_1y_all1000/
# Override via environment variable MINUTE_CSV_DIR if needed.
_DEFAULT_MINUTE_CSV_DIR = PROJECT_DIR / "leader_score_v2" / "data" / "minute_csv_1y_all1000"
MINUTE_CSV_DIR = Path(os.getenv("MINUTE_CSV_DIR", str(_DEFAULT_MINUTE_CSV_DIR)))
MINUTE_PARQUET_DIR = PROJECT_DIR / "data" / "minute"
EASTERN_TZ = "US/Eastern"
REGULAR_SESSION_START = pd.Timestamp("09:30").time()
REGULAR_SESSION_END = pd.Timestamp("16:00").time()

# Updated weights for momentum score optimization
WEIGHTS = {
    "RS2_Pct": 0.10,
    "RS3_Pct": 0.15,
    "RS5_Pct": 0.25,
    "RS10_Pct": 0.35,
    "RVOL_Pct": 0.15,
}


def load_minute_bars(ticker):
    csv_file = MINUTE_CSV_DIR / f"{ticker.lower()}_1min.csv"
    parquet_file = (
        MINUTE_PARQUET_DIR
        / ticker.lower()
        / f"{ticker.lower()}_1min.parquet"
    )

    if csv_file.exists():
        df = pd.read_csv(csv_file)
    elif parquet_file.exists():
        df = pd.read_parquet(parquet_file)
    else:
        raise FileNotFoundError(
            f"Minute data file not found for {ticker}: "
            f"{csv_file} or {parquet_file}"
        )

    required_cols = {"datetime", "open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Minute data for {ticker} must contain columns: "
            f"{sorted(required_cols)}"
        )

    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Massive aggregate timestamps are UTC. Convert to Eastern before
    # applying US regular-session filters or assigning the trading date.
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    else:
        df["datetime"] = df["datetime"].dt.tz_convert("UTC")

    df["datetime"] = df["datetime"].dt.tz_convert(EASTERN_TZ)

    return df


def filter_regular_session(df):
    df = df.copy()
    t = df["datetime"].dt.time

    return df[
        (t >= REGULAR_SESSION_START)
        & (t <= REGULAR_SESSION_END)
    ].copy()


def fetch_daily_bars(ticker, from_date, to_date):
    """
    Fetch daily bars by aggregating local minute data.
    
    Args:
        ticker (str): Stock ticker
        from_date (str): Start date in YYYY-MM-DD format
        to_date (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: OHLCV data with columns [Date, Open, High, Low, Close, Volume, VWAP]
    """
    minute_df = load_minute_bars(ticker)
    
    if minute_df.empty:
        return pd.DataFrame()
    
    minute_df = filter_regular_session(minute_df)
    
    if minute_df.empty:
        return pd.DataFrame()

    minute_df["date"] = minute_df["datetime"].dt.date
    
    # Filter date range
    from_dt = pd.to_datetime(from_date).date()
    to_dt = pd.to_datetime(to_date).date()
    minute_df = minute_df[(minute_df["date"] >= from_dt) & (minute_df["date"] <= to_dt)].copy()
    
    if minute_df.empty:
        return pd.DataFrame()
    
    # Aggregate to daily
    daily = minute_df.groupby("date").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).reset_index()
    
    # Calculate VWAP
    minute_df["typical_price"] = (minute_df["high"] + minute_df["low"] + minute_df["close"]) / 3
    minute_df["pv"] = minute_df["typical_price"] * minute_df["volume"]
    
    vwap = minute_df.groupby("date").agg({
        "pv": "sum",
        "volume": "sum"
    })
    vwap["vwap"] = vwap["pv"] / vwap["volume"]
    
    daily = daily.merge(vwap[["vwap"]], left_on="date", right_index=True, how="left")
    
    # Rename to match expected format
    daily = daily.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "vwap": "VWAP"
    })
    
    daily = daily.sort_values("Date").reset_index(drop=True)
    
    return daily[["Date", "Open", "High", "Low", "Close", "Volume", "VWAP"]]


def fetch_intraday_bars(ticker, target_date):
    """
    Fetch 1-minute intraday bars from local parquet file for a specific date.
    
    Args:
        ticker (str): Stock ticker symbol
        target_date (date): The date to fetch intraday data for (YYYY-MM-DD format or date object)
    
    Returns:
        pd.DataFrame: DataFrame with columns: datetime, open, high, low, close, volume
    """
    df = load_minute_bars(ticker)
    
    if df.empty:
        return pd.DataFrame()
    
    # Convert target_date to date object if needed
    if hasattr(target_date, 'strftime'):
        target_dt = target_date
    else:
        target_dt = pd.to_datetime(target_date).date()
    
    # Filter for specific date
    df["date"] = df["datetime"].dt.date
    df = df[df["date"] == target_dt].copy()
    
    # Drop the temporary date column
    df = df.drop(columns=["date"])
    
    return df[["datetime", "open", "high", "low", "close", "volume"]].copy()


def percentile_rank(series):
    return series.rank(pct=True) * 100


def safe_rs(stock_return, spy_return):
    if pd.isna(stock_return) or pd.isna(spy_return):
        return np.nan

    if spy_return == 0:
        return np.nan

    return (stock_return / spy_return) - 1


def build_leader_history(
    universe_csv="data/russell_1000.csv",
    output_csv="output/leader_events_1y.csv",
    threshold=70,
    lookback_days=365,
    warmup_days=90,
    sleep_seconds=0.15,
):
    if API_KEY is None:
        raise ValueError("Missing MASSIVE_API_KEY in project/.env file")

    universe = pd.read_csv(universe_csv)

    required_cols = {"ticker", "sector"}
    if not required_cols.issubset(universe.columns):
        raise ValueError("Universe CSV must contain: ticker, sector")

    today = datetime.now().date()

    # Need extra warmup for RS10 and 20-day RVOL
    fetch_from_date = today - timedelta(days=lookback_days + warmup_days)
    fetch_to_date = today

    fetch_from_str = fetch_from_date.strftime("%Y-%m-%d")
    fetch_to_str = fetch_to_date.strftime("%Y-%m-%d")

    final_cutoff_date = today - timedelta(days=lookback_days)

    print("=" * 60)
    print("ROLLING LEADER SCORE HISTORY (MOMENTUM-OPTIMIZED)")
    print("=" * 60)
    print(f"Fetching raw data from: {fetch_from_str} to {fetch_to_str}")
    print(f"Keeping events from:    {final_cutoff_date} to latest completed trading day")
    print(f"Leader threshold:      {threshold}")
    print(f"Universe file:         {universe_csv}")
    print(f"Weights: RS2={WEIGHTS['RS2_Pct']}, RS3={WEIGHTS['RS3_Pct']}, RS5={WEIGHTS['RS5_Pct']}, RS10={WEIGHTS['RS10_Pct']}, RVOL={WEIGHTS['RVOL_Pct']}")

    print("\nFetching SPY...")
    spy = fetch_daily_bars("SPY", fetch_from_str, fetch_to_str)

    if spy.empty or len(spy) < 80:
        raise ValueError("Not enough SPY data")

    spy = spy.sort_values("Date").reset_index(drop=True)

    spy["SPY_Ret2"] = spy["Close"].pct_change(2)
    spy["SPY_Ret3"] = spy["Close"].pct_change(3)
    spy["SPY_Ret5"] = spy["Close"].pct_change(5)
    spy["SPY_Ret10"] = spy["Close"].pct_change(10)

    latest_spy_date = spy["Date"].iloc[-1]
    print(f"Latest SPY trading date: {latest_spy_date}")

    all_stock_rows = []

    for i, row in universe.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        sector = str(row["sector"]).strip()

        if not ticker or ticker.lower() == "nan":
            continue

        try:
            print(f"[{i + 1}/{len(universe)}] Processing {ticker}")

            df = fetch_daily_bars(ticker, fetch_from_str, fetch_to_str)

            if df.empty or len(df) < 80:
                print(f"Skipping {ticker}: not enough daily data")
                continue

            df = df.sort_values("Date").reset_index(drop=True)

            df["Ticker"] = ticker
            df["Sector"] = sector

            # Rolling returns for every date
            df["Ret2"] = df["Close"].pct_change(2)
            df["Ret3"] = df["Close"].pct_change(3)
            df["Ret5"] = df["Close"].pct_change(5)
            df["Ret10"] = df["Close"].pct_change(10)

            # Rolling RVOL
            df["AvgVol20D"] = df["Volume"].rolling(20).mean()
            df["RVOL"] = df["Volume"] / df["AvgVol20D"]

            merged = df.merge(
                spy[
                    [
                        "Date",
                        "SPY_Ret2",
                        "SPY_Ret3",
                        "SPY_Ret5",
                        "SPY_Ret10",
                    ]
                ],
                on="Date",
                how="inner",
            )

            merged["RS2"] = [
                safe_rs(a, b)
                for a, b in zip(merged["Ret2"], merged["SPY_Ret2"])
            ]
            merged["RS3"] = [
                safe_rs(a, b)
                for a, b in zip(merged["Ret3"], merged["SPY_Ret3"])
            ]
            merged["RS5"] = [
                safe_rs(a, b)
                for a, b in zip(merged["Ret5"], merged["SPY_Ret5"])
            ]
            merged["RS10"] = [
                safe_rs(a, b)
                for a, b in zip(merged["Ret10"], merged["SPY_Ret10"])
            ]

            # Same liquidity filters as latest leader scanner
            merged = merged[
                (merged["Close"] >= 5)
                & (merged["AvgVol20D"] >= 500_000)
            ].copy()

            merged = merged.dropna(
                subset=["RS2", "RS3", "RS5", "RS10", "RVOL"]
            ).copy()

            if merged.empty:
                print(f"Skipping {ticker}: no valid rows after filters")
                continue

            all_stock_rows.append(
                merged[
                    [
                        "Date",
                        "Ticker",
                        "Sector",
                        "Close",
                        "Volume",
                        "AvgVol20D",
                        "RVOL",
                        "RS2",
                        "RS3",
                        "RS5",
                        "RS10",
                    ]
                ]
            )

            time.sleep(sleep_seconds)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    if not all_stock_rows:
        raise ValueError("No valid stock data found")

    data = pd.concat(all_stock_rows, ignore_index=True)

    data = data[data["Date"] >= final_cutoff_date].copy()

    if data.empty:
        raise ValueError("No rows after final 1-year cutoff")

    daily_outputs = []

    grouped_dates = sorted(data["Date"].unique())

    for date in grouped_dates:
        day_df = data[data["Date"] == date].copy()

        # Need enough names to percentile-rank properly
        if len(day_df) < 10:
            continue

        day_df["RS2_Pct"] = percentile_rank(day_df["RS2"])
        day_df["RS3_Pct"] = percentile_rank(day_df["RS3"])
        day_df["RS5_Pct"] = percentile_rank(day_df["RS5"])
        day_df["RS10_Pct"] = percentile_rank(day_df["RS10"])
        day_df["RVOL_Pct"] = percentile_rank(day_df["RVOL"])

        day_df["Leader_Score"] = (
            day_df["RS2_Pct"] * WEIGHTS["RS2_Pct"]
            + day_df["RS3_Pct"] * WEIGHTS["RS3_Pct"]
            + day_df["RS5_Pct"] * WEIGHTS["RS5_Pct"]
            + day_df["RS10_Pct"] * WEIGHTS["RS10_Pct"]
            + day_df["RVOL_Pct"] * WEIGHTS["RVOL_Pct"]
        )

        passed = day_df[day_df["Leader_Score"] >= threshold].copy()

        if not passed.empty:
            daily_outputs.append(passed)

    if not daily_outputs:
        print("No Leader Score events found.")
        return pd.DataFrame()

    events = pd.concat(daily_outputs, ignore_index=True)

    events = events.sort_values(
        ["Date", "Leader_Score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    output_cols = [
        "Date",
        "Ticker",
        "Sector",
        "Close",
        "Volume",
        "AvgVol20D",
        "RVOL",
        "RS2",
        "RS3",
        "RS5",
        "RS10",
        "RS2_Pct",
        "RS3_Pct",
        "RS5_Pct",
        "RS10_Pct",
        "RVOL_Pct",
        "Leader_Score",
    ]

    events = events[output_cols]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_csv, index=False)

    print("\nLeader events created:")
    print(events.head(30))
    print("\nTotal events:", len(events))
    print("Saved to:", output_csv)

    return events


if __name__ == "__main__":
    build_leader_history(
        universe_csv="data/russell_1000.csv",
        output_csv="output/leader_events_1y.csv",
        threshold=70,
        lookback_days=365,
        warmup_days=90,
        sleep_seconds=0.15,
    )
