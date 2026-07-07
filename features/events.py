"""
Event factors: earnings flags, technical events, pattern detection.

Earnings data workflow
----------------------
1. Run features/fetch_earnings.py to download from Massive API.
   Output: leader_score_v2/data/earnings_calendar.csv
   Columns: ticker, earnings_date, period_end_date, eps_actual

2. Load the CSV with load_earnings_calendar() — returns a dict
   mapping TICKER → sorted list of earnings dates.

3. Pass the per-ticker dates to compute_earnings_flags().
"""
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Earnings calendar loader
# ---------------------------------------------------------------------------

def load_earnings_calendar(
    csv_path: str | Path = None,
) -> dict[str, pd.DatetimeIndex]:
    """
    Load the pre-fetched earnings calendar CSV into a lookup dict.

    Args:
        csv_path: Path to earnings_calendar.csv produced by fetch_earnings.py.
                  Defaults to leader_score_v2/data/earnings_calendar.csv.

    Returns:
        Dict mapping ticker (upper-case str) → sorted DatetimeIndex of
        earnings dates.  Returns empty dict if file not found.
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "earnings_calendar.csv"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[events] Earnings calendar not found at {csv_path}. "
              "Run features/fetch_earnings.py first. Returning empty dict.")
        return {}

    df = pd.read_csv(csv_path, parse_dates=["earnings_date"])
    calendar: dict[str, pd.DatetimeIndex] = {}

    for ticker, group in df.groupby("ticker"):
        dates = pd.DatetimeIndex(sorted(group["earnings_date"].dropna()))
        calendar[ticker.upper().strip()] = dates

    print(f"[events] Loaded earnings calendar: {len(calendar)} tickers "
          f"({len(df)} records) from {csv_path.name}")
    return calendar


def compute_earnings_flags(
    dates: pd.Index,
    earnings_dates: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    Compute earnings proximity flag columns for a stock's date range.

    If earnings_dates is provided:
        DaysSinceEarnings     = calendar days since the most recent past earnings
        DaysUntilNextEarnings = calendar days until the next upcoming earnings
        RecentEarningsFlag    = True if DaysSinceEarnings <= 5
        UpcomingEarningsFlag  = True if DaysUntilNextEarnings <= 5

    If earnings_dates is None (no data integrated yet):
        All four columns are returned as NaN — placeholder for future API.

    Args:
        dates:          DatetimeIndex (or Index of dates) for the stock rows.
        earnings_dates: DatetimeIndex of known earnings announcement dates.
                        Pass None if earnings data is not yet available.

    Returns:
        DataFrame indexed by `dates` with columns:
            DaysSinceEarnings, DaysUntilNextEarnings,
            RecentEarningsFlag, UpcomingEarningsFlag
    """
    dates = pd.DatetimeIndex(dates)
    n = len(dates)

    # --- No earnings data yet — return placeholder NaNs ---
    if earnings_dates is None or len(earnings_dates) == 0:
        return pd.DataFrame(
            {
                "DaysSinceEarnings": np.nan,
                "DaysUntilNextEarnings": np.nan,
                "RecentEarningsFlag": np.nan,
                "UpcomingEarningsFlag": np.nan,
            },
            index=dates,
        )

    sorted_e = pd.DatetimeIndex(sorted(earnings_dates))

    days_since = np.empty(n, dtype=float)
    days_until = np.empty(n, dtype=float)
    days_since[:] = np.nan
    days_until[:] = np.nan

    for i, date in enumerate(dates):
        past = sorted_e[sorted_e <= date]
        future = sorted_e[sorted_e > date]
        if len(past) > 0:
            days_since[i] = (date - past[-1]).days
        if len(future) > 0:
            days_until[i] = (future[0] - date).days

    result = pd.DataFrame(
        {
            "DaysSinceEarnings": days_since,
            "DaysUntilNextEarnings": days_until,
        },
        index=dates,
    )
    result["RecentEarningsFlag"] = result["DaysSinceEarnings"] <= 5
    result["UpcomingEarningsFlag"] = result["DaysUntilNextEarnings"] <= 5

    return result


def detect_breakout(
    closes: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """
    Detect if today's close equals the rolling N-day high (new breakout).

    Args:
        closes:   Series of daily Close prices.
        lookback: Rolling window for breakout detection (default 252).

    Returns:
        Boolean Series — True on days where close == rolling max.
    """
    rolling_max = closes.rolling(lookback, min_periods=1).max()
    return closes >= rolling_max
