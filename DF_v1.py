"""
Defense First: A Multi-Asset Tactical Allocation Backtest
=========================================================
Paper: Thomas D. Carlson, July 1, 2025

Strategy: Rotates monthly among 4 defensive assets (TLT, GLD, DBC, UUP)
using equal-weighted momentum across 1/3/6/12-month lookbacks with a
40/30/20/10 tiered allocation. Absolute momentum filter vs. BIL (cash)
redirects weak slots to SPY as equity fallback.

"""

# ══════════════════════════════════════════════════════════════════════════
# §1  IMPORTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════
# %%

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ── Spyder plot settings ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"       : 120,
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#f8f8f8",
    "axes.grid"        : True,
    "grid.alpha"       : 0.4,
    "grid.linestyle"   : "--",
    "font.size"        : 10,
    "axes.titlesize"   : 11,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 9,
    "legend.fontsize"  : 9,
    "lines.linewidth"  : 1.6,
})

# ── Strategy parameters ───────────────────────────────────────────────────
DEFENSIVE_ASSETS = ["TLT", "GLD", "DBC", "UUP"]
FALLBACK_EQUITY  = "SPY"
CASH_PROXY       = "BIL"      # iShares 1-3 Month T-Bill ETF
BENCHMARK        = "SPY"

ALLOCATION_TIERS = {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10}
LOOKBACK_MONTHS  = [1, 3, 6, 12]   # momentum windows (equal-weighted)
TRADE_COST_BPS   = 25              # one-way cost per trade in basis points

START_DATE       = "2007-11-01"    # earliest date all ETFs are live
END_DATE         = "2026-03-31"            # None = through today

INITIAL_CAPITAL  = 10_000

# ── Output location ───────────────────────────────────────────────────────
import os
OUTPUT_DIR = os.path.expanduser("~/Documents")   # change if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────
C_STRAT  = "#1a6faf"   # Defense First — blue
C_BENCH  = "#555555"   # SPY benchmark — gray
C_DD     = "#c0392b"   # drawdown fill — red
ASSET_COLORS = {
    "TLT": "#2980b9",
    "GLD": "#f39c12",
    "DBC": "#27ae60",
    "UUP": "#8e44ad",
    "SPY": "#e74c3c",
}

print("§1  Config loaded.")
print(f"    Assets  : {DEFENSIVE_ASSETS}  |  Fallback: {FALLBACK_EQUITY}  |  Cash: {CASH_PROXY}")
print(f"    Lookbacks: {LOOKBACK_MONTHS} months  |  Tiers: {ALLOCATION_TIERS}")
print(f"    Period  : {START_DATE} → {'today' if END_DATE is None else END_DATE}")


# ══════════════════════════════════════════════════════════════════════════
# §2  DOWNLOAD DATA
# ══════════════════════════════════════════════════════════════════════════
# %%

ALL_TICKERS = list(dict.fromkeys(
    DEFENSIVE_ASSETS + [FALLBACK_EQUITY, CASH_PROXY, BENCHMARK]
))

print(f"Downloading: {ALL_TICKERS}  from {START_DATE} ...")
_dl = yf.download(
    ALL_TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False,
)

# yfinance >=0.2 returns MultiIndex columns: (field, ticker).
# Flatten to a simple ticker-keyed DataFrame regardless of version.
if isinstance(_dl.columns, pd.MultiIndex):
    _close_labels = [c for c in _dl.columns.get_level_values(0).unique()
                     if "close" in str(c).lower()]
    if not _close_labels:
        raise RuntimeError(
            f"No Close column found. Columns: {list(_dl.columns)}"
        )
    _raw = _dl[_close_labels[0]]
    if isinstance(_raw.columns, pd.MultiIndex):
        _raw.columns = _raw.columns.droplevel(0)
else:
    _raw = _dl["Close"] if "Close" in _dl.columns else _dl

if isinstance(_raw, pd.Series):
    _raw = _raw.to_frame(ALL_TICKERS[0])

prices = _raw.ffill()

# Validate
_missing = [t for t in ALL_TICKERS if t not in prices.columns]
if _missing:
    print(f"  WARNING — could not download: {_missing}")

_required = DEFENSIVE_ASSETS + [FALLBACK_EQUITY, CASH_PROXY]
_bad = [t for t in _required if t not in prices.columns or prices[t].isna().all()]
if _bad:
    raise RuntimeError(f"Required tickers missing or empty: {_bad}\n"
                       "Check internet connection and try again.")

print(f"  OK — {len(prices)} daily bars  "
      f"({prices.index[0].date()} → {prices.index[-1].date()})")
print(f"  Columns : {list(prices.columns)}")
print(f"  yfinance: {yf.__version__}  |  pandas: {pd.__version__}")

# Sanity check: warn if we got suspiciously few bars
_expected_min_bars = 200
if len(prices) < _expected_min_bars:
    print(f"\n  *** WARNING: Only {len(prices)} bars downloaded — expected ~"
          f"{_expected_min_bars}+ for a multi-year backtest.")
    print(f"      This usually means yfinance returned a MultiIndex that was")
    print(f"      not fully flattened.  Raw _dl columns: {list(_dl.columns)[:8]}")
    print(f"      Try: pip install --upgrade yfinance  then restart kernel.\n")


# ══════════════════════════════════════════════════════════════════════════
# §3  RUN BACKTEST
# ══════════════════════════════════════════════════════════════════════════
# %%

def _momentum_score(monthly_series):
    """Equal-weighted avg of 1/3/6/12-month simple returns."""
    parts = [monthly_series.pct_change(lb) for lb in LOOKBACK_MONTHS]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False)


def run_backtest(daily_prices):
    """
    For each month-end:
      1. Score each defensive asset via composite momentum.
      2. Rank highest → lowest; assign 40/30/20/10 weights.
      3. Absolute filter: if score < cash score, redirect that slot to SPY.
      4. Record forward 1-month return.

    Returns
    -------
    returns_df     : DataFrame with columns [strategy, benchmark], monthly freq
    allocations_df : DataFrame of monthly weights per ticker
    scores_df      : DataFrame of raw monthly momentum scores
    """
    # "ME" = month-end in pandas >=2.2; fall back to "M" for older versions
    try:
        monthly = daily_prices.resample("ME").last()
    except ValueError:
        monthly = daily_prices.resample("M").last()

    # ── momentum scores ───────────────────────────────────────────────
    def_scores  = {t: _momentum_score(monthly[t]) for t in DEFENSIVE_ASSETS}
    score_df    = pd.DataFrame(def_scores)
    cash_scores = _momentum_score(monthly[CASH_PROXY])

    strat_rets, bench_rets = [], []
    alloc_rows, score_rows = [], []

    dates = monthly.index[13:]   # need 12-month lookback to warm up

    for date in dates:
        loc      = monthly.index.get_loc(date)
        next_loc = loc + 1
        if next_loc >= len(monthly):
            break
        next_date = monthly.index[next_loc]

        row_scores = score_df.loc[date]
        cash_val   = cash_scores.loc[date]

        if row_scores.isna().any() or pd.isna(cash_val):
            continue

        # ── rank & apply absolute momentum filter ─────────────────────
        ranked  = row_scores.sort_values(ascending=False)
        weights = {}
        for rank_idx, (ticker, score) in enumerate(ranked.items(), start=1):
            tier_wt = ALLOCATION_TIERS[rank_idx]
            if score >= cash_val:
                weights[ticker] = weights.get(ticker, 0) + tier_wt
            else:
                weights[FALLBACK_EQUITY] = weights.get(FALLBACK_EQUITY, 0) + tier_wt

        # ── forward 1-month return ────────────────────────────────────
        port_ret  = sum(
            w * (monthly.loc[next_date, t] / monthly.loc[date, t] - 1)
            for t, w in weights.items()
            if t in monthly.columns
        )
        bench_ret = monthly.loc[next_date, BENCHMARK] / monthly.loc[date, BENCHMARK] - 1

        strat_rets.append(port_ret)
        bench_rets.append(bench_ret)

        # log full weight vector for every ticker
        all_row = {t: weights.get(t, 0.0) for t in ALL_TICKERS}
        alloc_rows.append({"date": next_date, **all_row})
        score_rows.append({"date": next_date, **row_scores.to_dict(),
                           "BIL": cash_val})

    if not alloc_rows:
        n_monthly = len(monthly)
        raise RuntimeError(
            f"Backtest produced no trades.\n"
            f"  Monthly bars available : {n_monthly}  (need >13 to warm up)\n"
            f"  Check that START_DATE is early enough and data downloaded correctly.\n"
            f"  prices shape: {daily_prices.shape}, "
            f"date range: {daily_prices.index[0].date()} to {daily_prices.index[-1].date()}"
        )

    idx = pd.DatetimeIndex([r["date"] for r in alloc_rows])

    returns_df = pd.DataFrame(
        {"strategy": strat_rets, "benchmark": bench_rets}, index=idx
    ).dropna()

    allocations_df = pd.DataFrame(alloc_rows).set_index("date")
    scores_df      = pd.DataFrame(score_rows).set_index("date")

    return returns_df, allocations_df, scores_df


returns_df, allocations_df, scores_df = run_backtest(prices)

# ── equity curves & drawdowns ─────────────────────────────────────────────
strat_curve = (1 + returns_df["strategy"]).cumprod()  * INITIAL_CAPITAL
bench_curve = (1 + returns_df["benchmark"]).cumprod() * INITIAL_CAPITAL

def _drawdown(ret_series):
    cum  = (1 + ret_series).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak

strat_dd = _drawdown(returns_df["strategy"])
bench_dd = _drawdown(returns_df["benchmark"])

print(f"§3  Backtest complete — {len(returns_df)} monthly observations "
      f"({returns_df.index[0].strftime('%Y-%m')} → {returns_df.index[-1].strftime('%Y-%m')})")


# ══════════════════════════════════════════════════════════════════════════
# §4  PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════
# %%

def calc_metrics(ret, name, rf=0.02):
    n        = len(ret)
    years    = n / 12
    cagr     = (1 + ret).prod() ** (1 / years) - 1
    vol      = ret.std() * np.sqrt(12)
    sharpe   = (cagr - rf) / vol if vol else 0
    rf_m     = (1 + rf) ** (1/12) - 1
    downside = ret[ret < rf_m].std() * np.sqrt(12)
    sortino  = (cagr - rf) / downside if downside else 0
    cum      = (1 + ret).cumprod()
    max_dd   = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar   = cagr / abs(max_dd) if max_dd else 0
    win_pct  = (ret > 0).mean() * 100
    annual   = ret.groupby(ret.index.year).apply(lambda r: (1+r).prod()-1)
    beta     = ret.cov(returns_df["benchmark"]) / returns_df["benchmark"].var()
    corr     = ret.corr(returns_df["benchmark"])
    return {
        "Name"         : name,
        "CAGR"         : cagr,
        "Volatility"   : vol,
        "Sharpe Ratio" : sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown" : max_dd,
        "Calmar Ratio" : calmar,
        "Best Year"    : annual.max(),
        "Worst Year"   : annual.min(),
        "Win Rate %"   : win_pct,
        "Beta vs SPY"  : beta,
        "Corr vs SPY"  : corr,
        "Months"       : n,
    }

sm = calc_metrics(returns_df["strategy"],  "Defense First")
bm = calc_metrics(returns_df["benchmark"], "SPY")

# ── trading cost estimate ─────────────────────────────────────────────────
_monthly_cost   = 0.20 * TRADE_COST_BPS / 10_000   # ~20% avg monthly turnover
_adj_rets       = returns_df["strategy"] - _monthly_cost
_years          = len(_adj_rets) / 12
_cagr_adj       = (1 + _adj_rets).prod() ** (1 / _years) - 1
_annual_cost_dr = _monthly_cost * 12

# ── print table ───────────────────────────────────────────────────────────
FMT = {
    "CAGR"         : "{:.2%}",
    "Volatility"   : "{:.2%}",
    "Sharpe Ratio" : "{:.2f}",
    "Sortino Ratio": "{:.2f}",
    "Max Drawdown" : "{:.2%}",
    "Calmar Ratio" : "{:.2f}",
    "Best Year"    : "{:.2%}",
    "Worst Year"   : "{:.2%}",
    "Win Rate %"   : "{:.1f}%",
    "Beta vs SPY"  : "{:.2f}",
    "Corr vs SPY"  : "{:.2f}",
    "Months"       : "{:d}",
}

_w = 16
print(f"\n{'═'*55}")
print("  PERFORMANCE SUMMARY")
print(f"{'═'*55}")
print(f"  {'Metric':<18}  {'Defense First':>{_w}}  {'SPY':>{_w}}")
print(f"  {'─'*18}  {'─'*_w}  {'─'*_w}")
for k, fmt in FMT.items():
    sv = fmt.format(sm[k]) if k != "Months" else str(sm[k])
    bv = fmt.format(bm[k]) if k != "Months" else str(bm[k])
    print(f"  {k:<18}  {sv:>{_w}}  {bv:>{_w}}")

print(f"\n  ── Trading Cost Estimate (25bps/trade, ~20% monthly turnover) ──")
print(f"  CAGR before costs : {sm['CAGR']:.2%}")
print(f"  CAGR after costs  : {_cagr_adj:.2%}")
print(f"  Annual cost drag  : {_annual_cost_dr:.2%}")
print(f"{'═'*55}\n")


# ══════════════════════════════════════════════════════════════════════════
# §5  ANNUAL RETURNS
# ══════════════════════════════════════════════════════════════════════════
# %%

ann_strat = (returns_df["strategy"]
             .groupby(returns_df.index.year)
             .apply(lambda r: (1+r).prod()-1))

ann_bench = (returns_df["benchmark"]
             .groupby(returns_df.index.year)
             .apply(lambda r: (1+r).prod()-1))

annual_df = pd.DataFrame({
    "Defense First": ann_strat,
    "SPY"          : ann_bench,
    "Difference"   : ann_strat - ann_bench,
})
annual_df.index.name = "Year"

print("ANNUAL RETURNS\n")
print(f"  {'Year':>4}  {'Defense First':>14}  {'SPY':>10}  {'Diff':>8}")
print(f"  {'─'*4}  {'─'*14}  {'─'*10}  {'─'*8}")
for yr, row in annual_df.iterrows():
    diff_sign = "+" if row["Difference"] > 0 else ""
    print(f"  {yr:>4}  {row['Defense First']:>14.2%}  "
          f"{row['SPY']:>10.2%}  {diff_sign}{row['Difference']:>7.2%}")

# also available in variable explorer as: annual_df


# ══════════════════════════════════════════════════════════════════════════
# §6  CRISIS PERIOD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
# %%

CRISIS_PERIODS = {
    "GFC 2007-09"          : ("2007-11", "2009-03"),
    "COVID Crash Q1 2020"  : ("2020-01", "2020-03"),
    "Inflation Shock 2022" : ("2022-01", "2022-12"),
}

crisis_rows = []
for label, (s, e) in CRISIS_PERIODS.items():
    try:
        w = returns_df.loc[s:e]
        if len(w) == 0:
            continue
        strat = (1 + w["strategy"]).prod() - 1
        bench = (1 + w["benchmark"]).prod() - 1
        crisis_rows.append({
            "Crisis Period"  : label,
            "Defense First"  : strat,
            "SPY"            : bench,
            "Outperformance" : strat - bench,
        })
    except Exception:
        pass

crisis_df = pd.DataFrame(crisis_rows).set_index("Crisis Period")

print("CRISIS PERIOD PERFORMANCE\n")
print(f"  {'Period':<26}  {'Defense First':>14}  {'SPY':>10}  {'Alpha':>8}")
print(f"  {'─'*26}  {'─'*14}  {'─'*10}  {'─'*8}")
for lbl, row in crisis_df.iterrows():
    sign = "+" if row["Outperformance"] > 0 else ""
    print(f"  {lbl:<26}  {row['Defense First']:>14.2%}  "
          f"{row['SPY']:>10.2%}  {sign}{row['Outperformance']:>7.2%}")


# ══════════════════════════════════════════════════════════════════════════
# §7  EQUITY CURVE CHART
# ══════════════════════════════════════════════════════════════════════════
# %%

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(strat_curve, color=C_STRAT, label="Defense First", linewidth=2)
ax.plot(bench_curve, color=C_BENCH, label="SPY (benchmark)",
        linewidth=1.5, alpha=0.75, linestyle="--")

ax.set_yscale("log")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
)

# annotate end values
for curve, color, name in [
    (strat_curve, C_STRAT, "Defense First"),
    (bench_curve, C_BENCH, "SPY"),
]:
    ax.annotate(
        f"${curve.iloc[-1]:,.0f}",
        xy=(curve.index[-1], curve.iloc[-1]),
        xytext=(8, 0), textcoords="offset points",
        fontsize=8, color=color, va="center",
    )

ax.set_title(
    f"Defense First vs. SPY — Growth of ${INITIAL_CAPITAL:,}  "
    f"({returns_df.index[0].strftime('%b %Y')} – "
    f"{returns_df.index[-1].strftime('%b %Y')})"
)
ax.set_xlabel("")
ax.set_ylabel("Portfolio Value (log scale)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "defense_first_equity_curve.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("§7  Chart saved: defense_first_equity_curve.png")


# ══════════════════════════════════════════════════════════════════════════
# §8  DRAWDOWN CHART
# ══════════════════════════════════════════════════════════════════════════
# %%

fig, ax = plt.subplots(figsize=(11, 4))

ax.fill_between(strat_dd.index, strat_dd * 100, 0,
                color=C_STRAT, alpha=0.30, label="Defense First")
ax.fill_between(bench_dd.index, bench_dd * 100, 0,
                color=C_DD, alpha=0.20, label="SPY")

ax.plot(strat_dd * 100, color=C_STRAT, linewidth=1.2)
ax.plot(bench_dd * 100, color=C_DD, linewidth=1.0, linestyle="--", alpha=0.7)

ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_title("Drawdown History")
ax.set_xlabel("")
ax.set_ylabel("Drawdown (%)")
ax.legend(loc="lower right")

# annotate max drawdowns
for dd_series, color in [(strat_dd, C_STRAT), (bench_dd, C_DD)]:
    idx_min = dd_series.idxmin()
    ax.annotate(
        f"{dd_series.min():.1%}",
        xy=(idx_min, dd_series.min() * 100),
        xytext=(0, -14), textcoords="offset points",
        fontsize=8, color=color, ha="center",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "defense_first_drawdowns.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("§8  Chart saved: defense_first_drawdowns.png")


# ══════════════════════════════════════════════════════════════════════════
# §9  MONTHLY RETURNS HEATMAP
# ══════════════════════════════════════════════════════════════════════════
# %%

# pivot to year × month grid
monthly_ret = returns_df["strategy"].copy()
heatmap_df  = pd.DataFrame({
    "Year" : monthly_ret.index.year,
    "Month": monthly_ret.index.month,
    "Ret"  : monthly_ret.values,
}).pivot(index="Year", columns="Month", values="Ret")

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
heatmap_df.columns = [MONTH_NAMES[m-1] for m in heatmap_df.columns]

# symmetric colormap centered at 0
_vmax = max(abs(heatmap_df.values[~np.isnan(heatmap_df.values)]).max(), 0.05)
cmap = LinearSegmentedColormap.from_list(
    "rg", ["#c0392b", "#ffffff", "#1a6faf"]
)

fig, ax = plt.subplots(figsize=(13, max(4, len(heatmap_df) * 0.38 + 1)))

im = ax.imshow(heatmap_df.values, cmap=cmap,
               vmin=-_vmax, vmax=_vmax, aspect="auto")

ax.set_xticks(range(len(heatmap_df.columns)))
ax.set_xticklabels(heatmap_df.columns, fontsize=8)
ax.set_yticks(range(len(heatmap_df.index)))
ax.set_yticklabels(heatmap_df.index, fontsize=8)

# cell annotations
for i in range(len(heatmap_df.index)):
    for j in range(len(heatmap_df.columns)):
        val = heatmap_df.values[i, j]
        if not np.isnan(val):
            txt   = f"{val:.1%}".replace("-", "−")
            color = "white" if abs(val) > _vmax * 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6.5, color=color)

# annual totals on right axis
ann_vals = heatmap_df.apply(
    lambda row: (1 + row.dropna()).prod() - 1, axis=1
)
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(ann_vals)))
ax2.set_yticklabels(
    [f"{v:.1%}" for v in ann_vals],
    fontsize=7,
    color=C_STRAT,
)
ax2.set_ylabel("Annual Total", color=C_STRAT, fontsize=8)

plt.colorbar(im, ax=ax, fraction=0.02, pad=0.12,
             format=mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_title("Defense First — Monthly Returns Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "defense_first_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("§9  Chart saved: defense_first_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
# §10 ASSET ALLOCATION OVER TIME  (stacked area)
# ══════════════════════════════════════════════════════════════════════════
# %%

# only plot columns with non-zero allocations
alloc_plot = allocations_df[[
    t for t in DEFENSIVE_ASSETS + [FALLBACK_EQUITY]
    if t in allocations_df.columns
]]

fig, ax = plt.subplots(figsize=(11, 4))

ax.stackplot(
    alloc_plot.index,
    [alloc_plot[t] for t in alloc_plot.columns],
    labels=list(alloc_plot.columns),
    colors=[ASSET_COLORS.get(t, "#aaaaaa") for t in alloc_plot.columns],
    alpha=0.80,
)

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_title("Defense First — Monthly Asset Allocation")
ax.set_ylabel("Portfolio Weight")
ax.legend(loc="upper left", ncol=len(alloc_plot.columns), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "defense_first_allocation.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("§10 Chart saved: defense_first_allocation.png")


# ══════════════════════════════════════════════════════════════════════════
# §11 CURRENT ALLOCATION  (latest month)
# ══════════════════════════════════════════════════════════════════════════
# %%

latest_alloc = allocations_df.iloc[-1]
latest_alloc = latest_alloc[latest_alloc > 0].sort_values(ascending=False)
latest_date  = allocations_df.index[-1].strftime("%B %Y")

fig, ax = plt.subplots(figsize=(5, 3))
bars = ax.barh(
    latest_alloc.index[::-1],
    latest_alloc.values[::-1],
    color=[ASSET_COLORS.get(t, "#aaaaaa") for t in latest_alloc.index[::-1]],
    edgecolor="white", linewidth=0.5,
)

for bar, val in zip(bars, latest_alloc.values[::-1]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.0%}", va="center", fontsize=9)

ax.set_xlim(0, 0.55)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.set_title(f"Current Allocation — {latest_date}", pad=8)
ax.set_xlabel("Portfolio Weight")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "defense_first_current_alloc.png"),
            dpi=150, bbox_inches="tight")
plt.show()

print(f"\nCurrent Allocation  ({latest_date})")
print("─" * 28)
for ticker, w in latest_alloc.items():
    bar = "█" * int(w * 40)
    print(f"  {ticker:<5} {bar:<18}  {w:.0%}")


# ══════════════════════════════════════════════════════════════════════════
# §12 SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════
# %%

_ret_path   = os.path.join(OUTPUT_DIR, "defense_first_returns.csv")
_alloc_path = os.path.join(OUTPUT_DIR, "defense_first_allocations.csv")
_scores_path= os.path.join(OUTPUT_DIR, "defense_first_scores.csv")
_annual_path= os.path.join(OUTPUT_DIR, "defense_first_annual.csv")

returns_df.to_csv(_ret_path)
allocations_df.to_csv(_alloc_path)
scores_df.to_csv(_scores_path)
annual_df.to_csv(_annual_path)

print("§12 Results saved:")
print(f"    {_ret_path}")
print(f"    {_alloc_path}")
print(f"    {_scores_path}")
print(f"    {_annual_path}")
print("\nAll done. Variables available in Explorer:")
print("  prices, returns_df, allocations_df, scores_df,")
print("  strat_curve, bench_curve, strat_dd, bench_dd,")
print("  annual_df, crisis_df, latest_alloc, sm, bm")


# ══════════════════════════════════════════════════════════════════════════
# §13  COMPARISON TO PAPER (Carlson, July 2025)
# ══════════════════════════════════════════════════════════════════════════
# %%
# Compares annual returns from this backtest against Appendix A of the paper.
# Only years present in both datasets are shown.
# Paper's 2025 figure covers Jan-Jun 2025 only (published July 1 2025).

PAPER_ANNUAL = {
    1986: (18.80, 18.06), 1987: (10.59,  4.71), 1988: (12.66, 16.22),
    1989: (18.64, 31.36), 1990: (-1.46, -3.32), 1991: (24.96, 30.22),
    1992: ( 5.37,  7.42), 1993: ( 9.25,  9.89), 1994: (-6.94,  1.18),
    1995: (27.46, 37.45), 1996: (16.34, 22.88), 1997: (20.30, 33.19),
    1998: (24.40, 28.62), 1999: (10.88, 21.07), 2000: ( 7.78, -9.06),
    2001: (-6.73,-12.02), 2002: ( 8.59,-22.15), 2003: (12.90, 28.50),
    2004: (15.87, 10.74), 2005: (16.89,  4.77), 2006: (11.57, 15.64),
    2007: (13.96,  5.39), 2008: ( 5.69,-37.02), 2009: ( 7.36, 26.49),
    2010: (11.56, 14.91), 2011: (12.95,  1.97), 2012: (-2.00, 15.82),
    2013: (16.32, 32.18), 2014: ( 8.83, 13.51), 2015: (-1.67,  1.25),
    2016: ( 7.89, 11.82), 2017: ( 4.82, 21.67), 2018: (-3.17, -4.52),
    2019: (11.99, 31.33), 2020: (20.10, 18.25), 2021: (23.15, 28.53),
    2022: ( 8.07,-18.23), 2023: ( 7.00, 26.11), 2024: (17.02, 24.84),
    2025: (13.92,  6.13),   # paper = H1 2025 only (pub. July 1 2025)
}

# ── Build comparison DataFrame ────────────────────────────────────────────
my_years = annual_df.index.tolist()
paper_years = sorted(PAPER_ANNUAL.keys())
overlap = sorted(set(my_years) & set(paper_years))

comp_rows = []
for yr in overlap:
    my_df_val  = annual_df.loc[yr, "Defense First"] * 100
    my_spy_val = annual_df.loc[yr, "SPY"]           * 100
    paper_df, paper_spy = PAPER_ANNUAL[yr]
    diff       = my_df_val - paper_df
    note       = ""
    if yr == 2025:
        note = "paper=H1 only"
    elif abs(diff) < 0.05:
        note = "exact"
    elif abs(diff) < 0.30:
        note = "near exact"
    comp_rows.append({
        "Year"         : yr,
        "Mine (DF)"    : my_df_val,
        "Paper (DF)"   : paper_df,
        "Diff"         : diff,
        "Mine (SPY)"   : my_spy_val,
        "Paper (SPY)"  : paper_spy,
        "Note"         : note,
    })

comp_df = pd.DataFrame(comp_rows).set_index("Year")

# ── Print comparison table ────────────────────────────────────────────────
_w = 65
print(f"\n{'═'*_w}")
print("  COMPARISON TO PAPER  (Carlson, July 2025 — Appendix A)")
print(f"{'═'*_w}")
print(f"  {'Year':>4}  {'Mine':>8}  {'Paper':>8}  {'Diff':>8}  "
      f"{'My SPY':>8}  {'Ppr SPY':>8}  {'Note'}")
print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  "
      f"{'─'*8}  {'─'*8}  {'─'*10}")

for yr, row in comp_df.iterrows():
    diff_sign = "+" if row["Diff"] >= 0 else ""
    print(f"  {yr:>4}  "
          f"{row['Mine (DF)']:>+8.2f}%  "
          f"{row['Paper (DF)']:>+8.2f}%  "
          f"{diff_sign}{row['Diff']:>7.2f}%  "
          f"{row['Mine (SPY)']:>+8.2f}%  "
          f"{row['Paper (SPY)']:>+8.2f}%  "
          f"{row['Note']}")

# ── Summary statistics ────────────────────────────────────────────────────
diffs        = comp_df["Diff"].abs()
exact        = (comp_df["Note"] == "exact").sum()
near_exact   = (comp_df["Note"] == "near exact").sum()
same_dir     = ((comp_df["Mine (DF)"] * comp_df["Paper (DF)"]) > 0).sum()
avg_abs_diff = diffs.mean()
max_diff_yr  = diffs.idxmax()

print(f"\n{'─'*_w}")
print(f"  Overlap years          : {len(overlap)}")
print(f"  Exact matches (±0.05%) : {exact}")
print(f"  Near exact  (±0.30%)   : {near_exact}")
print(f"  Same direction         : {same_dir} / {len(overlap)}")
print(f"  Avg absolute diff      : {avg_abs_diff:.2f}%")
print(f"  Largest diff           : {diffs.max():.2f}% in {max_diff_yr}"
      f"  ({comp_df.loc[max_diff_yr, 'Note']})")
print(f"{'═'*_w}")

# comp_df is available in Variable Explorer for further inspection