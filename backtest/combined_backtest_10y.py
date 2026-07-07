"""
10-Year Combined Long Portfolio Backtest
=========================================

Same strategy as combined_backtest.py but runs on the 10y scored dataset
and 10y all-sessions minute data.

Key differences from 1y version:
  - Data: scored_dataset_10y.csv  +  minute_csv_10y_all1000/
  - Minute file naming: {ticker}_1min_all_sessions.csv
  - Timestamps in 10y files are Eastern time (not UTC)
    → 3:30 PM ET bar is '15:30'  (was '19:30' UTC in 1y files)
    → regular hours filter uses session == 'regular' column
  - Enhanced trade output: Entry_Price, Exit_Price, SL_Price, SL_Pct,
    Stop_Triggered, Exit_Reason per trade
  - Saves: combined_trades_10y.csv + combined_equity_10y.csv

Usage:
    python leader_score_v2/backtest/combined_backtest_10y.py
    python leader_score_v2/backtest/combined_backtest_10y.py --no_neutral_below
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[1]
SCORED     = ROOT / "output" / "scored_dataset_10y.csv"
MINUTE_DIR = ROOT / "data" / "minute_csv_10y_all1000"
OUT        = ROOT / "output" / "backtest"

WEIGHTS = {
    "RS10_Pct": 0.45, "RS5_Pct": 0.25,
    "RVOL_Pct": 0.20, "RS3_Pct": 0.05, "DollarVolume_Pct": 0.05,
}

BULL_MIN_SCORE    = 80.0
NEUTRAL_MIN_SCORE = 85.0
TOP_N             = 10
HOLD              = 5
ATR_STOP_MULT     = 2.0
TRAIL_STOP        = True   # True = ATR trailing stop  |  False = fixed ATR stop


# ---------------------------------------------------------------------------
def add_raw_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Raw_Score"]    = sum(df[c].fillna(50) * w for c, w in WEIGHTS.items())
    df["Above_SMA50"]  = df["SPY_Close"] >= df["SPY_SMA50"]
    df["Above_SMA200"] = df["SPY_Close"] >= df["SPY_SMA200"]
    return df


def build_trades(df: pd.DataFrame, include_neutral_below: bool = True) -> pd.DataFrame:
    rows = []
    for date, day_df in df.groupby("Date"):
        bull      = day_df["SPY_Bull"].iloc[0] == True
        above50   = day_df["Above_SMA50"].iloc[0]
        above200  = day_df["Above_SMA200"].iloc[0]

        if bull:
            regime, min_score = "Bull", BULL_MIN_SCORE
        elif above50:
            regime, min_score = "Neutral_Above", NEUTRAL_MIN_SCORE
        else:
            if not include_neutral_below:
                continue
            # Only trade Neutral_Below when SPY is still above 200-day MA
            # (below SMA50 but above SMA200 = short-term pullback, not a bear market)
            if not above200:
                continue
            regime, min_score = "Neutral_Below", NEUTRAL_MIN_SCORE

        eligible = day_df[
            (day_df["Raw_Score"] >= min_score)
            & day_df["FwdRet_5D"].notna()
            & np.isfinite(day_df["FwdRet_5D"])
        ]
        top = eligible.nlargest(TOP_N, "Raw_Score")
        if top.empty:
            continue

        for _, r in top.iterrows():
            rows.append({
                "Date":      date,
                "Ticker":    r["Ticker"],
                "Sector":    r.get("Sector", ""),
                "Regime":    regime,
                "Raw_Score": r["Raw_Score"],
                "RS10_Pct":  r.get("RS10_Pct", np.nan),
                "RVOL_Pct":  r.get("RVOL_Pct", np.nan),
                "Close":     r["Close"],
                "ATR20":     r.get("ATR20", np.nan),
                "Return":    r["FwdRet_5D"],
            })
    return pd.DataFrame(rows)


def build_equity(df: pd.DataFrame, trades_df: pd.DataFrame,
                 include_neutral_below: bool = True) -> pd.DataFrame:
    spy_close  = df.groupby("Date")["SPY_Close"].first().sort_index()
    all_dates  = sorted(spy_close.index)
    active_dates = sorted(trades_df["Date"].unique())

    rebal_dates, last_idx = [], -HOLD
    for i, d in enumerate(active_dates):
        if i - last_idx >= HOLD:
            rebal_dates.append(d)
            last_idx = i

    portfolio_value = 1.0
    spy_value       = 1.0
    rows = []

    for date in rebal_dates:
        period_trades = trades_df[trades_df["Date"] == date]
        if period_trades.empty:
            continue

        period_ret = period_trades["Return"].mean()
        regime     = period_trades["Regime"].iloc[0]

        future = [d for d in all_dates if d > date]
        spy_ret = (spy_close[future[HOLD-1]] - spy_close[date]) / spy_close[date] if len(future) >= HOLD else 0.0

        portfolio_value *= (1 + period_ret)
        spy_value       *= (1 + spy_ret)

        rows.append({
            "Date":              date,
            "Regime":            regime,
            "Portfolio_Value":   portfolio_value,
            "SPY_Value":         spy_value,
            "Period_Return":     period_ret,
            "SPY_Period_Return": spy_ret,
            "N_Stocks":          len(period_trades),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single-pass minute data loader:
#   - 3:30 PM close (entry price)
#   - daily min low  (stop check)
#   - daily max high (trailing stop ratchet)
#   - trading calendar
# Reads each file exactly ONCE to avoid 2x I/O overhead.
# ---------------------------------------------------------------------------
def _preload_all_minute_data(tickers):
    t0                = time.time()
    price_330         = {}        # (date_str, ticker) -> 3:30 PM close
    daily_min_lows    = {}        # ticker -> {date_str: min_low}
    daily_max_highs   = {}        # ticker -> {date_str: max_high}
    trading_dates_set = set()

    n = len(tickers)
    print(f"\nLoading minute data (single pass) for {n} tickers...")
    print(f"  [3:30 PM entry prices + daily low/high for ATR stops]")
    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0 or i == n:
            print(f"  {i}/{n} ({time.time()-t0:.0f}s elapsed)...")
        csv_path = MINUTE_DIR / f"{ticker.lower()}_1min_all_sessions.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(
            csv_path,
            usecols=["datetime", "session", "close", "low", "high"],
            dtype={"datetime": str, "session": str,
                   "close": float, "low": float, "high": float},
        )
        df = df[df["session"] == "regular"].copy()
        df["date"] = df["datetime"].str[:10]

        # 3:30 PM close
        df330 = df[df["datetime"].str[11:16] == "15:30"]
        for d, p in zip(df330["date"], df330["close"]):
            price_330[(d, ticker)] = p

        # daily min/max
        min_low  = df.groupby("date")["low"].min()
        max_high = df.groupby("date")["high"].max()
        daily_min_lows[ticker]  = min_low.to_dict()
        daily_max_highs[ticker] = max_high.to_dict()
        trading_dates_set.update(min_low.index)

    trading_dates = sorted(trading_dates_set)
    td_index      = {d: i for i, d in enumerate(trading_dates)}
    trading_cal   = {d: trading_dates[i + 1: i + 1 + HOLD] for d, i in td_index.items()}

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s) | {len(price_330):,} price points | {len(trading_dates)} trading days")
    return price_330, daily_min_lows, daily_max_highs, trading_cal


def _adjust_entry_to_330(trades: pd.DataFrame, price_330: dict) -> pd.DataFrame:
    trades   = trades.copy()
    adjusted = 0
    for idx, row in trades.iterrows():
        key  = (str(row["Date"])[:10], row["Ticker"])
        p330 = price_330.get(key)
        if p330 is None or p330 <= 0:
            continue
        close_t  = row["Close"]
        close_t5 = close_t * (1 + row["Return"])     # Close[t+5]
        trades.at[idx, "Return"] = (close_t5 - p330) / p330
        trades.at[idx, "Close"]  = p330               # entry price for stop calc
        adjusted += 1
    pct = adjusted / len(trades) * 100 if len(trades) else 0
    print(f"  3:30 PM entry adjusted: {adjusted}/{len(trades)} trades ({pct:.1f}%)")
    return trades


# ---------------------------------------------------------------------------
# ATR stop loss  — supports fixed and trailing modes
# ---------------------------------------------------------------------------
def apply_atr_stops(
    trades: pd.DataFrame,
    entry_lookup: pd.Series,
    atr_lookup: pd.Series,
    daily_min_lows: dict,
    daily_max_highs: dict,
    trading_cal: dict,
) -> pd.DataFrame:
    """Apply 2x ATR stop loss (fixed or trailing based on TRAIL_STOP flag).
    Returns copy with Return, SL_Price, Exit_Price, Stop_Triggered, Exit_Reason."""
    trades = trades.copy()

    mode = "trailing" if TRAIL_STOP else "fixed"
    print(f"  Stop mode: {mode} ATR ({ATR_STOP_MULT}x)")

    new_returns     = {}
    sl_prices       = {}   # always = initial fixed stop (entry - 2xATR)
    exit_prices     = {}
    stop_triggered  = {}
    exit_reasons    = {}
    stopped = 0

    for idx, row in trades.iterrows():
        ticker     = row["Ticker"]
        trade_date = str(row["Date"])[:10]
        orig       = row["Return"]
        key        = (row["Date"], ticker)

        ticker_lows  = daily_min_lows.get(ticker)
        ticker_highs = daily_max_highs.get(ticker)
        if ticker_lows is None:
            new_returns[idx]    = orig
            sl_prices[idx]      = np.nan
            exit_prices[idx]    = np.nan
            stop_triggered[idx] = False
            exit_reasons[idx]   = "Held_t5"
            continue

        try:
            entry_close = entry_lookup.loc[key]
            atr         = atr_lookup.loc[key]
        except KeyError:
            new_returns[idx]    = orig
            sl_prices[idx]      = np.nan
            exit_prices[idx]    = np.nan
            stop_triggered[idx] = False
            exit_reasons[idx]   = "Held_t5"
            continue

        if pd.isna(atr) or atr <= 0 or pd.isna(entry_close) or entry_close <= 0:
            new_returns[idx]    = orig
            sl_prices[idx]      = np.nan
            exit_prices[idx]    = round(entry_close * (1 + orig), 4) if entry_close > 0 else np.nan
            stop_triggered[idx] = False
            exit_reasons[idx]   = "Held_t5"
            continue

        risk          = atr * ATR_STOP_MULT
        initial_stop  = entry_close - risk
        if initial_stop <= 0:
            new_returns[idx]    = orig
            sl_prices[idx]      = np.nan
            exit_prices[idx]    = round(entry_close * (1 + orig), 4)
            stop_triggered[idx] = False
            exit_reasons[idx]   = "Held_t5"
            continue

        result     = orig
        hit_stop   = False
        trail_stop = initial_stop   # start at fixed stop
        trail_ref  = entry_close    # highest price seen so far

        for hd in trading_cal.get(trade_date, []):
            # 1) Check today's low against stop set by PREVIOUS day's ratchet
            day_low = ticker_lows.get(hd, float("inf"))
            if day_low <= trail_stop:
                result   = (trail_stop - entry_close) / entry_close
                hit_stop = True
                stopped += 1
                break

            # 2) Ratchet stop up based on today's high (applies from NEXT day)
            if TRAIL_STOP and ticker_highs is not None:
                day_high = ticker_highs.get(hd, 0.0)
                if day_high > trail_ref:
                    trail_ref  = day_high
                    new_stop   = trail_ref - risk
                    trail_stop = max(trail_stop, new_stop)   # ratchet up only

        new_returns[idx]    = result
        sl_prices[idx]      = round(initial_stop, 4)          # always initial stop
        exit_prices[idx]    = round(entry_close * (1 + result), 4)
        stop_triggered[idx] = hit_stop
        exit_reasons[idx]   = ("Trail_Stop" if TRAIL_STOP else "Stop") if hit_stop else "Held_t5"

    trades["Return"]        = pd.Series(new_returns)
    trades["SL_Price"]      = pd.Series(sl_prices)
    trades["Exit_Price"]    = pd.Series(exit_prices)
    trades["Stop_Triggered"]= pd.Series(stop_triggered)
    trades["Exit_Reason"]   = pd.Series(exit_reasons)

    pct_stopped = stopped / len(trades) * 100 if len(trades) else 0
    print(f"  Stops triggered: {stopped}/{len(trades)} ({pct_stopped:.1f}%)")
    return trades


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(trades: pd.DataFrame, equity: pd.DataFrame, label: str):
    n         = len(trades)
    avg_ret   = trades["Return"].mean() * 100
    win_rate  = (trades["Return"] > 0).mean() * 100
    wins      = trades[trades["Return"] > 0]["Return"]
    losses    = trades[trades["Return"] <= 0]["Return"]
    avg_win   = wins.mean() * 100 if len(wins) else 0.0
    avg_loss  = losses.mean() * 100 if len(losses) else 0.0
    rr        = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    total     = (equity["Portfolio_Value"].iloc[-1] - 1) * 100
    spy_tot   = (equity["SPY_Value"].iloc[-1] - 1) * 100

    rets   = equity["Period_Return"]
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252 / HOLD) if rets.std() > 0 else 0
    max_dd = ((equity["Portfolio_Value"] / equity["Portfolio_Value"].cummax()) - 1).min() * 100

    n_years = (pd.to_datetime(equity["Date"].max()) - pd.to_datetime(equity["Date"].min())).days / 365.25
    cagr    = ((equity["Portfolio_Value"].iloc[-1]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Trades        : {n:,}  |  Active days: {trades['Date'].nunique()}")
    print(f"  Win rate      : {win_rate:.1f}%  |  Avg return: {avg_ret:+.2f}%")
    print(f"  Avg win       : {avg_win:+.2f}%  |  Avg loss: {avg_loss:+.2f}%  |  R/R: {rr:.2f}")
    print(f"  ---")
    print(f"  Total return  : {total:+.1f}%")
    print(f"  CAGR          : {cagr:+.1f}%")
    print(f"  SPY total     : {spy_tot:+.1f}%")
    print(f"  Alpha         : {total-spy_tot:+.1f}%")
    print(f"  Sharpe        : {sharpe:.2f}")
    print(f"  Max drawdown  : {max_dd:.1f}%")
    print(f"{'='*65}")

    print(f"\n  BY REGIME:")
    print(f"  {'Regime':<22} {'Days':>5} {'Trades':>7} {'Avg Ret':>9} {'Win Rate':>9}")
    print(f"  {'-'*57}")
    for regime, grp in trades.groupby("Regime"):
        days = grp["Date"].nunique()
        avg  = grp["Return"].mean() * 100
        wr   = (grp["Return"] > 0).mean() * 100
        print(f"  {regime:<22} {days:>5} {len(grp):>7} {avg:>+8.2f}% {wr:>8.1f}%")
    print()


# ---------------------------------------------------------------------------
def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_neutral_below", action="store_true",
                        help="Skip neutral-below-SMA50 days")
    args = parser.parse_args()

    if not SCORED.exists():
        raise FileNotFoundError(
            f"Scored dataset not found: {SCORED}\n"
            "Run: python leader_score_v2/backtest/build_scored_dataset_10y.py"
        )

    print(f"\nLoading {SCORED.name} ...")
    df = pd.read_csv(SCORED, low_memory=False, parse_dates=["Date"])
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    print(f"Loaded : {len(df):,} rows | {df['Date'].nunique()} dates | {df['Ticker'].nunique()} tickers")

    df = add_raw_score(df)

    # --- Build trades ---
    trades = build_trades(df, include_neutral_below=not args.no_neutral_below)

    # --- Adjust entry to 3:30 PM ET price + preload minute data (single pass) ---
    price_330, daily_min_lows, daily_max_highs, trading_cal = \
        _preload_all_minute_data(trades["Ticker"].unique())
    trades = _adjust_entry_to_330(trades, price_330)

    # --- Apply 2.0× ATR stop losses ---
    entry_lookup = trades.set_index(["Date", "Ticker"])["Close"]
    atr_lookup   = (
        df[["Date", "Ticker", "ATR20"]].drop_duplicates(["Date", "Ticker"])
        .set_index(["Date", "Ticker"])["ATR20"]
    )
    trades = apply_atr_stops(trades, entry_lookup, atr_lookup, daily_min_lows, daily_max_highs, trading_cal)

    # Rename Close → Entry_Price for clarity in output
    trades = trades.rename(columns={"Close": "Entry_Price"})

    # Compute SL_Pct
    trades["SL_Pct"] = np.where(
        trades["SL_Price"].notna() & (trades["Entry_Price"] > 0),
        (trades["SL_Price"] - trades["Entry_Price"]) / trades["Entry_Price"] * 100,
        np.nan,
    )

    # --- Equity curve ---
    equity = build_equity(df, trades, include_neutral_below=not args.no_neutral_below)

    label = f"10Y COMBINED: Bull + Neutral [3:30 PM entry, {ATR_STOP_MULT}x ATR {'trailing' if TRAIL_STOP else 'fixed'} stop]"
    print_summary(trades, equity, label)

    # --- Save ---
    OUT.mkdir(parents=True, exist_ok=True)

    trades_out = trades[[
        "Date", "Ticker", "Sector", "Regime", "Raw_Score",
        "RS10_Pct", "RVOL_Pct", "ATR20",
        "Entry_Price", "Exit_Price", "SL_Price", "SL_Pct",
        "Stop_Triggered", "Exit_Reason", "Return",
    ]].copy()
    trades_out["Return_Pct"] = (trades_out["Return"] * 100).round(4)
    trades_out["Year"]       = pd.to_datetime(trades_out["Date"]).dt.year

    trades_csv = OUT / "combined_trades_10y.csv"
    equity_csv = OUT / "combined_equity_10y.csv"
    trades_out.to_csv(trades_csv, index=False)
    equity.to_csv(equity_csv, index=False)
    print(f"\nSaved → {trades_csv}  ({len(trades_out):,} trades)")
    print(f"Saved → {equity_csv}  ({len(equity)} periods)")
    print("\nNext step: python leader_score_v2/backtest/build_10y_report.py")


if __name__ == "__main__":
    _main()
