"""
Defense First — ETF Analysis Suite
=====================================
Combines two analyses into one script. Run DF_v1.py first, then this file.

  PART A — ETF Contribution Analysis  (§1 – §10)
    How many months each ETF is held, portfolio exposure, and P&L attribution.

  PART B — Momentum Signal Predictive Power  (§11 – §16)
    Does the BIL momentum filter actually predict forward returns for each ETF?

DEPENDENCIES
  Requires returns_df, allocations_df, prices to already be in the Spyder
  namespace — run DF_v1.py first, then run this script.

  Alternatively, set STANDALONE = True in §1 to download data fresh.

SPYDER USAGE
  Run cells with Ctrl+Enter or full file with F5.
  All results land in Variable Explorer as DataFrames.
"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART A — ETF CONTRIBUTION ANALYSIS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════════════
# §1  IMPORTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════
# %%

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Set to True to download data fresh (ignores existing namespace) ────────
STANDALONE   = True
START_DATE   = "2007-06-01"
END_DATE     = "2026-03-31"
OUTPUT_DIR   = os.path.expanduser("~/Documents")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_ETFS     = ["TLT", "GLD", "DBC", "UUP", "BIL", "SPY"]
TICKER_LABELS = {
    "TLT": "TLT (LT Treasuries)",
    "GLD": "GLD (Gold)",
    "DBC": "DBC (Commodities)",
    "UUP": "UUP (US Dollar)",
    "BIL": "BIL (Cash / T-Bills)",
    "SPY": "SPY (Equity Fallback)",
}

COLORS = {
    "TLT": "#2980b9",
    "GLD": "#f39c12",
    "DBC": "#27ae60",
    "UUP": "#8e44ad",
    "BIL": "#7f8c8d",
    "SPY": "#e74c3c",
}

# Signal state colors (Part B momentum charts)
C_POS  = "#1D9E75"   # green  — positive momentum
C_NEG  = "#E24B4A"   # red    — negative momentum
C_ABOV = "#2980b9"   # blue   — above BIL
C_BELO = "#e67e22"   # orange — below BIL

sns.set_theme(style="whitegrid", font_scale=0.9)

plt.rcParams.update({
    "figure.dpi"      : 120,
    "figure.facecolor": "white",
    "axes.titlesize"  : 10,
    "axes.titleweight": "bold",
    "axes.labelsize"  : 9,
    "legend.fontsize" : 8,
    "lines.linewidth" : 1.4,
})

print("§1  Config loaded.")


# ══════════════════════════════════════════════════════════════════════════
# §2  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
# %%

if STANDALONE:
    import yfinance as yf

    DEFENSIVE_ASSETS = ["TLT", "GLD", "DBC", "UUP"]
    FALLBACK_EQUITY  = "SPY"
    CASH_PROXY       = "BIL"
    BENCHMARK        = "SPY"
    ALLOCATION_TIERS = {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10}
    LOOKBACK_MONTHS  = [1, 3, 6, 12]
    INITIAL_CAPITAL  = 10_000

    print(f"Downloading data ...")
    _dl = yf.download(ALL_ETFS, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    if isinstance(_dl.columns, pd.MultiIndex):
        _close = [c for c in _dl.columns.get_level_values(0).unique()
                  if "close" in str(c).lower()]
        _raw = _dl[_close[0]]
        if isinstance(_raw.columns, pd.MultiIndex):
            _raw.columns = _raw.columns.droplevel(0)
    else:
        _raw = _dl["Close"] if "Close" in _dl.columns else _dl
    prices = _raw.ffill()

    try:
        monthly = prices.resample("ME").last()
    except ValueError:
        monthly = prices.resample("M").last()

    def _mom_score(s):
        parts = [s.pct_change(lb) for lb in LOOKBACK_MONTHS]
        return pd.concat(parts, axis=1).mean(axis=1, skipna=False)

    def run_backtest(daily_prices):
        try:
            mo = daily_prices.resample("ME").last()
        except ValueError:
            mo = daily_prices.resample("M").last()

        def_scores  = {t: _mom_score(mo[t]) for t in DEFENSIVE_ASSETS}
        score_df    = pd.DataFrame(def_scores)
        cash_scores = _mom_score(mo[CASH_PROXY])

        strat_rets, bench_rets, alloc_rows = [], [], []
        for date in mo.index[13:]:
            loc = mo.index.get_loc(date)
            if loc + 1 >= len(mo): break
            next_date  = mo.index[loc + 1]
            row_scores = score_df.loc[date]
            cash_val   = cash_scores.loc[date]
            if row_scores.isna().any() or pd.isna(cash_val): continue
            ranked  = row_scores.sort_values(ascending=False)
            weights = {}
            for rank_idx, (ticker, score) in enumerate(ranked.items(), start=1):
                tier_wt = ALLOCATION_TIERS[rank_idx]
                if score >= cash_val:
                    weights[ticker] = weights.get(ticker, 0) + tier_wt
                else:
                    weights[FALLBACK_EQUITY] = weights.get(FALLBACK_EQUITY, 0) + tier_wt
            port_ret  = sum(w * (mo.loc[next_date, t] / mo.loc[date, t] - 1)
                            for t, w in weights.items() if t in mo.columns)
            bench_ret = mo.loc[next_date, BENCHMARK] / mo.loc[date, BENCHMARK] - 1
            strat_rets.append(port_ret)
            bench_rets.append(bench_ret)
            all_row = {t: weights.get(t, 0.0) for t in ALL_ETFS}
            alloc_rows.append({"date": next_date, **all_row})

        idx = pd.DatetimeIndex([r["date"] for r in alloc_rows])
        returns_df    = pd.DataFrame({"strategy": strat_rets,
                                       "benchmark": bench_rets}, index=idx).dropna()
        allocations_df = pd.DataFrame(alloc_rows).set_index("date")
        return returns_df, allocations_df

    returns_df, allocations_df = run_backtest(prices)
    print(f"  Backtest complete: {len(returns_df)} months")

else:
    # Use existing namespace from DF_v1.py
    try:
        _ = returns_df, allocations_df, prices
        print(f"  Using existing backtest: {len(returns_df)} monthly observations")
        print(f"  ({returns_df.index[0].strftime('%Y-%m')} → "
              f"{returns_df.index[-1].strftime('%Y-%m')})")
    except NameError:
        raise RuntimeError(
            "returns_df / allocations_df not found in namespace.\n"
            "Run DF_v1.py first, or set STANDALONE = True in §1."
        )

# Align allocations to returns index (drop current-month signal row)
alloc = allocations_df.reindex(returns_df.index).fillna(0)

# Ensure all ETFs are present as columns
for t in ALL_ETFS:
    if t not in alloc.columns:
        alloc[t] = 0.0

print(f"  Allocation matrix: {alloc.shape}  "
      f"({alloc.index[0].strftime('%Y-%m')} → {alloc.index[-1].strftime('%Y-%m')})")


# ══════════════════════════════════════════════════════════════════════════
# §3  COMPUTE MONTHLY P&L ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
# %%

# Monthly prices aligned to returns index
try:
    monthly = prices.resample("ME").last()
except ValueError:
    monthly = prices.resample("M").last()

# Forward returns for each ETF each month (same period as strategy returns)
etf_fwd_rets = monthly[ALL_ETFS].pct_change().reindex(returns_df.index)

# Monthly P&L contribution per ETF = weight × ETF return that month
# This gives the exact dollar contribution as a fraction of portfolio value
pnl_contrib = alloc[ALL_ETFS].multiply(etf_fwd_rets[ALL_ETFS])

print(f"§3  P&L attribution computed: {pnl_contrib.shape}")
print(f"    Row sums match strategy returns: "
      f"{'YES' if (pnl_contrib.sum(axis=1).round(6) == returns_df['strategy'].round(6)).all() else 'CHECK'}")


# ══════════════════════════════════════════════════════════════════════════
# §4  ANALYSIS 1 — MONTHS HELD
# ══════════════════════════════════════════════════════════════════════════
# %%

# A month is "held" if the allocation weight > 0
months_held   = (alloc[ALL_ETFS] > 0).sum()
total_months  = len(alloc)
pct_months    = months_held / total_months * 100

months_df = pd.DataFrame({
    "ETF"           : [TICKER_LABELS[t] for t in ALL_ETFS],
    "Months Held"   : months_held[ALL_ETFS].values,
    "% of Months"   : pct_months[ALL_ETFS].values,
}, index=ALL_ETFS)

print(f"\n{'═'*60}")
print(f"  ANALYSIS 1 — MONTHS HELD  (total backtest: {total_months} months)")
print(f"{'═'*60}")
print(f"\n  {'Ticker':>5}  {'Months':>7}  {'% of Months':>12}")
print(f"  {'─'*5}  {'─'*7}  {'─'*12}")
for t in ALL_ETFS:
    print(f"  {t:>5}  {months_df.loc[t,'Months Held']:>7.0f}  "
          f"{months_df.loc[t,'% of Months']:>11.1f}%")
print(f"\n  Note: months held sum to more than {total_months} because")
print(f"  multiple ETFs are held simultaneously each month.")


# ══════════════════════════════════════════════════════════════════════════
# §5  ANALYSIS 2 — PORTFOLIO EXPOSURE
# ══════════════════════════════════════════════════════════════════════════
# %%

# Total exposure = sum of (weight × 1) across all months
# Normalised so all ETFs sum to 100%
total_exposure   = alloc[ALL_ETFS].sum()         # sum of weights across all months
pct_exposure     = total_exposure / total_exposure.sum() * 100
avg_weight_when_held = pd.Series({
    t: alloc.loc[alloc[t] > 0, t].mean() * 100
    for t in ALL_ETFS
})

exposure_df = pd.DataFrame({
    "ETF"                : [TICKER_LABELS[t] for t in ALL_ETFS],
    "Total Weight-Months": total_exposure[ALL_ETFS].values,
    "% of Exposure"      : pct_exposure[ALL_ETFS].values,
    "Avg Weight (held)"  : avg_weight_when_held[ALL_ETFS].values,
}, index=ALL_ETFS)

print(f"\n{'═'*68}")
print(f"  ANALYSIS 2 — PORTFOLIO EXPOSURE")
print(f"{'═'*68}")
print(f"\n  {'Ticker':>5}  {'Wt-Months':>10}  {'% Exposure':>11}  "
      f"{'Avg Wt (when held)':>19}")
print(f"  {'─'*5}  {'─'*10}  {'─'*11}  {'─'*19}")
for t in ALL_ETFS:
    print(f"  {t:>5}  "
          f"{exposure_df.loc[t,'Total Weight-Months']:>10.1f}  "
          f"{exposure_df.loc[t,'% of Exposure']:>10.1f}%  "
          f"{exposure_df.loc[t,'Avg Weight (held)']:>18.1f}%")
print(f"\n  % Exposure = (sum of monthly weights) / (total of all weights)")
print(f"  Avg Wt     = average weight in months when the ETF is held")


# ══════════════════════════════════════════════════════════════════════════
# §6  ANALYSIS 3 — P&L ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════
# %%

total_pnl_per_etf  = pnl_contrib[ALL_ETFS].sum()        # cumulative contribution
total_strategy_pnl = returns_df["strategy"].sum()        # total strategy P&L (sum of monthly rets)

pct_pnl           = total_pnl_per_etf / abs(total_strategy_pnl) * 100
avg_monthly_contrib = pnl_contrib[ALL_ETFS].mean() * 100

# Cumulative P&L curve per ETF
cum_pnl = pnl_contrib[ALL_ETFS].cumsum()

pnl_df = pd.DataFrame({
    "ETF"                : [TICKER_LABELS[t] for t in ALL_ETFS],
    "Cumulative Contrib" : total_pnl_per_etf[ALL_ETFS].values,
    "% of Total P&L"     : pct_pnl[ALL_ETFS].values,
    "Avg Monthly Contrib": avg_monthly_contrib[ALL_ETFS].values,
}, index=ALL_ETFS)

print(f"\n{'═'*72}")
print(f"  ANALYSIS 3 — P&L ATTRIBUTION")
print(f"  Total strategy return (sum of monthly): {total_strategy_pnl:.4f} "
      f"({total_strategy_pnl*100:.2f}%)")
print(f"{'═'*72}")
print(f"\n  {'Ticker':>5}  {'Cum Contrib':>12}  {'% of P&L':>10}  "
      f"{'Avg Monthly':>12}  {'Direction'}")
print(f"  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*10}")
for t in ALL_ETFS:
    cum   = pnl_df.loc[t, "Cumulative Contrib"]
    pct   = pnl_df.loc[t, "% of Total P&L"]
    avg   = pnl_df.loc[t, "Avg Monthly Contrib"]
    direc = "Positive" if cum > 0 else "Negative"
    print(f"  {t:>5}  {cum:>+12.4f}  {pct:>+9.1f}%  "
          f"{avg:>+11.3f}%  {direc}")

print(f"\n  % of P&L = ETF cumulative contribution / |total strategy P&L|")
print(f"  Values can exceed 100% / be negative due to offsetting contributions.")


# ══════════════════════════════════════════════════════════════════════════
# §7  COMBINED SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
# %%

contribution_summary = pd.DataFrame({
    "ETF"               : [TICKER_LABELS[t] for t in ALL_ETFS],
    "Months Held"       : months_held[ALL_ETFS].values.astype(int),
    "% of Months"       : pct_months[ALL_ETFS].round(1).values,
    "% Exposure"        : pct_exposure[ALL_ETFS].round(1).values,
    "Avg Wt (held)"     : avg_weight_when_held[ALL_ETFS].round(1).values,
    "% of P&L"          : pct_pnl[ALL_ETFS].round(1).values,
    "Avg Mthly Contrib" : avg_monthly_contrib[ALL_ETFS].round(3).values,
}, index=ALL_ETFS)

print(f"\n{'═'*90}")
print(f"  COMBINED SUMMARY")
print(f"{'═'*90}")
print(f"\n  {'Ticker':>5}  {'Mo Held':>7}  {'%Mo':>5}  "
      f"{'%Exp':>6}  {'AvgWt':>6}  {'%P&L':>7}  {'AvgMo%':>8}")
print(f"  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*8}")
for t in ALL_ETFS:
    r = contribution_summary.loc[t]
    print(f"  {t:>5}  {r['Months Held']:>7}  {r['% of Months']:>4.1f}%  "
          f"{r['% Exposure']:>5.1f}%  {r['Avg Wt (held)']:>5.1f}%  "
          f"{r['% of P&L']:>+6.1f}%  {r['Avg Mthly Contrib']:>+7.3f}%")


# ══════════════════════════════════════════════════════════════════════════
# §8  CHARTS
# ══════════════════════════════════════════════════════════════════════════
# %%

period_str = (f"{returns_df.index[0].strftime('%b %Y')} – "
              f"{returns_df.index[-1].strftime('%b %Y')}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

bar_colors = [COLORS[t] for t in ALL_ETFS]
labels     = [t for t in ALL_ETFS]

# ── Chart 1: Months Held ──────────────────────────────────────────────────
ax = axes[0]
bars = sns.barplot(
    x=labels,
    y=months_held[ALL_ETFS].values,
    palette=bar_colors,
    edgecolor="white", linewidth=0.8,
    alpha=0.88, ax=ax,
)
for patch, val in zip(ax.patches, months_held[ALL_ETFS].values):
    ax.text(patch.get_x() + patch.get_width()/2,
            val + 1,
            f"{val:.0f}\n({val/total_months*100:.0f}%)",
            ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_title("Months Held", pad=8)
ax.set_xlabel("")
ax.set_ylabel("Number of months")
ax.set_ylim(0, max(months_held[ALL_ETFS].values) * 1.25)

# ── Chart 2: Portfolio Exposure ───────────────────────────────────────────
ax = axes[1]
sns.barplot(
    x=labels,
    y=pct_exposure[ALL_ETFS].values,
    palette=bar_colors,
    edgecolor="white", linewidth=0.8,
    alpha=0.88, ax=ax,
)
for patch, val in zip(ax.patches, pct_exposure[ALL_ETFS].values):
    ax.text(patch.get_x() + patch.get_width()/2,
            val + 0.3,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_title("Portfolio Exposure", pad=8)
ax.set_xlabel("")
ax.set_ylabel("% of total exposure")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0f}%")
)
ax.set_ylim(0, max(pct_exposure[ALL_ETFS].values) * 1.25)

# ── Chart 3: P&L Attribution ─────────────────────────────────────────────
ax = axes[2]
pnl_vals = pct_pnl[ALL_ETFS].values
bar_clrs  = [COLORS[t] if pnl_vals[i] >= 0 else "#cccccc"
             for i, t in enumerate(ALL_ETFS)]
sns.barplot(
    x=labels,
    y=pnl_vals,
    palette=bar_clrs,
    edgecolor="white", linewidth=0.8,
    alpha=0.88, ax=ax,
)
for patch, val in zip(ax.patches, pnl_vals):
    yoff = 0.5 if val >= 0 else -2.5
    ax.text(patch.get_x() + patch.get_width()/2,
            val + yoff,
            f"{val:+.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8.5, fontweight="bold")

ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
ax.set_title("% of Total P&L", pad=8)
ax.set_xlabel("")
ax.set_ylabel("% of strategy P&L")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%")
)

fig.suptitle(
    f"Defense First — ETF Contribution Analysis  |  {period_str}",
    fontsize=11, fontweight="bold", y=1.02,
)
plt.tight_layout()
_chart1 = os.path.join(OUTPUT_DIR, "df_etf_contribution.png")
plt.savefig(_chart1, dpi=150, bbox_inches="tight")
plt.show()
print(f"§8  Chart saved: {_chart1}")


# ══════════════════════════════════════════════════════════════════════════
# §9  CUMULATIVE P&L CONTRIBUTION OVER TIME
# ══════════════════════════════════════════════════════════════════════════
# %%

fig, ax = plt.subplots(figsize=(12, 5))

for t in ALL_ETFS:
    cum = cum_pnl[t] * 100
    ax.plot(cum.index, cum, label=t, color=COLORS[t],
            linewidth=1.5, alpha=0.85)
    # End-of-line label
    ax.annotate(
        f"{t} ({cum.iloc[-1]:+.1f}%)",
        xy=(cum.index[-1], cum.iloc[-1]),
        xytext=(6, 0), textcoords="offset points",
        fontsize=7.5, color=COLORS[t], va="center",
    )

ax.axhline(0, color="black", linewidth=0.8, alpha=0.5, linestyle="--")
ax.set_title(
    f"Cumulative P&L Contribution by ETF  |  {period_str}",
    fontsize=11, fontweight="bold", pad=8,
)
ax.set_ylabel("Cumulative contribution to strategy return (%)")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:+.0f}%")
)
ax.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
sns.despine()
plt.tight_layout()
_chart2 = os.path.join(OUTPUT_DIR, "df_etf_cumulative_pnl.png")
plt.savefig(_chart2, dpi=150, bbox_inches="tight")
plt.show()
print(f"§9  Chart saved: {_chart2}")


# ══════════════════════════════════════════════════════════════════════════
# §10 SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════
# %%

_summary_path = os.path.join(OUTPUT_DIR, "df_etf_contribution_summary.csv")
_pnl_path     = os.path.join(OUTPUT_DIR, "df_etf_pnl_monthly.csv")
_cum_path     = os.path.join(OUTPUT_DIR, "df_etf_pnl_cumulative.csv")

contribution_summary.to_csv(_summary_path)
pnl_contrib.to_csv(_pnl_path)
(cum_pnl * 100).to_csv(_cum_path)

print(f"\n§10 Results saved:")
print(f"    {_summary_path}")
print(f"    {_pnl_path}")
print(f"    {_cum_path}")
print(f"\nVariables in Explorer:")
print(f"  contribution_summary — combined metrics per ETF")
print(f"  months_df     — months held analysis")
print(f"  exposure_df   — portfolio exposure analysis")
print(f"  pnl_df        — P&L attribution analysis")
print(f"  pnl_contrib   — monthly P&L contribution per ETF (matrix)")
print(f"  cum_pnl       — cumulative P&L contribution per ETF")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART B — MOMENTUM SIGNAL PREDICTIVE POWER                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ══════════════════════════════════════════════════════════════════════════
# §11 COMPUTE MOMENTUM SCORES & FORWARD RETURNS
# ══════════════════════════════════════════════════════════════════════════
# %%

# ── Composite momentum score for each asset ───────────────────────────────
def composite_score(series):
    parts = [series.pct_change(lb) for lb in LOOKBACK_MONTHS]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False)

scores = pd.DataFrame({t: composite_score(monthly[t]) for t in ALL_ETFS})

# ── Forward 1-month return for each asset ────────────────────────────────
fwd_returns = monthly.pct_change().shift(-1)   # next month's return

# ── Build analysis DataFrame ─────────────────────────────────────────────
# Drop warmup period and last row (no forward return available)
valid_idx = scores.dropna().index[:-1]

analysis_rows = []
for date in valid_idx:
    for ticker in DEFENSIVE_ASSETS:
        score     = scores.loc[date, ticker]
        bil_score = scores.loc[date, CASH_PROXY]
        fwd_ret   = fwd_returns.loc[date, ticker]

        if pd.isna(score) or pd.isna(fwd_ret):
            continue

        analysis_rows.append({
            "date"           : date,
            "ticker"         : ticker,
            "score"          : score,
            "bil_score"      : bil_score,
            "fwd_return"     : fwd_ret,
            # Signal definitions
            "pos_vs_zero"    : score > 0,        # composite score > 0
            "pos_vs_bil"     : score > bil_score, # composite score > BIL
            "score_quartile" : None,              # filled below
        })

analysis_df = pd.DataFrame(analysis_rows)

# Add quartile classification per ticker
for ticker in DEFENSIVE_ASSETS:
    mask = analysis_df["ticker"] == ticker
    analysis_df.loc[mask, "score_quartile"] = pd.qcut(
        analysis_df.loc[mask, "score"],
        q=4, labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"]
    )

print(f"§11 Analysis DataFrame: {len(analysis_df)} rows  "
      f"({analysis_df['date'].min().strftime('%Y-%m')} → "
      f"{analysis_df['date'].max().strftime('%Y-%m')})")


# ══════════════════════════════════════════════════════════════════════════
# §12 SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
# %%

print(f"\n{'═'*75}")
print("  MOMENTUM SIGNAL PREDICTIVE POWER — AVERAGE FORWARD 1-MONTH RETURN")
print(f"{'═'*75}")

summary_rows = []
for ticker in DEFENSIVE_ASSETS:
    sub = analysis_df[analysis_df["ticker"] == ticker]

    # vs zero
    pos_zero = sub[sub["pos_vs_zero"]]["fwd_return"]
    neg_zero = sub[~sub["pos_vs_zero"]]["fwd_return"]

    # vs BIL
    pos_bil  = sub[sub["pos_vs_bil"]]["fwd_return"]
    neg_bil  = sub[~sub["pos_vs_bil"]]["fwd_return"]

    row = {
        "Ticker"          : ticker,
        "N_total"         : len(sub),
        # vs zero
        "N_pos_zero"      : len(pos_zero),
        "N_neg_zero"      : len(neg_zero),
        "Ret_pos_zero"    : pos_zero.mean(),
        "Ret_neg_zero"    : neg_zero.mean(),
        "Spread_zero"     : pos_zero.mean() - neg_zero.mean(),
        # vs BIL
        "N_pos_bil"       : len(pos_bil),
        "N_neg_bil"       : len(neg_bil),
        "Ret_pos_bil"     : pos_bil.mean(),
        "Ret_neg_bil"     : neg_bil.mean(),
        "Spread_bil"      : pos_bil.mean() - neg_bil.mean(),
        # unconditional
        "Ret_all"         : sub["fwd_return"].mean(),
    }
    summary_rows.append(row)

mom_summary_df = pd.DataFrame(summary_rows).set_index("Ticker")

# Print table — vs zero
print(f"\n  Signal: Momentum score > 0  (positive vs negative)")
print(f"  {'Ticker':>6}  {'N(+)':>5} {'Avg Ret(+)':>11} "
      f"{'N(-)':>5} {'Avg Ret(-)':>11} {'Spread':>9} {'Unconditional':>14}")
print(f"  {'─'*6}  {'─'*5} {'─'*11} {'─'*5} {'─'*11} {'─'*9} {'─'*14}")
for t, row in mom_summary_df.iterrows():
    print(f"  {t:>6}  "
          f"{row['N_pos_zero']:>5.0f} "
          f"{row['Ret_pos_zero']:>+11.3%} "
          f"{row['N_neg_zero']:>5.0f} "
          f"{row['Ret_neg_zero']:>+11.3%} "
          f"{row['Spread_zero']:>+9.3%} "
          f"{row['Ret_all']:>+14.3%}")

# Print table — vs BIL
print(f"\n  Signal: Momentum score > BIL  (above vs below cash)")
print(f"  {'Ticker':>6}  {'N(>BIL)':>7} {'Avg Ret':>9} "
      f"{'N(<BIL)':>7} {'Avg Ret':>9} {'Spread':>9}")
print(f"  {'─'*6}  {'─'*7} {'─'*9} {'─'*7} {'─'*9} {'─'*9}")
for t, row in mom_summary_df.iterrows():
    print(f"  {t:>6}  "
          f"{row['N_pos_bil']:>7.0f} "
          f"{row['Ret_pos_bil']:>+9.3%} "
          f"{row['N_neg_bil']:>7.0f} "
          f"{row['Ret_neg_bil']:>+9.3%} "
          f"{row['Spread_bil']:>+9.3%}")


# ══════════════════════════════════════════════════════════════════════════
# §13 BAR CHARTS — ONE PER ETF (2×2 grid, seaborn) — BIL filter only
# ══════════════════════════════════════════════════════════════════════════
# %%

# Long-form DataFrame: BIL filter signal only (the actual strategy signal)
plot_rows = []
for ticker in DEFENSIVE_ASSETS:
    sub = analysis_df[analysis_df["ticker"] == ticker]
    for signal_label, mask in [
        ("Score > BIL (above cash)", sub["pos_vs_bil"]),        ("Score ≤ BIL(below cash)", ~sub["pos_vs_bil"]),
    ]:
        ret = sub.loc[mask, "fwd_return"].mean() * 100
        n   = int(mask.sum())
        plot_rows.append({
            "ticker"  : ticker,
            "signal"  : signal_label,
            "ret_pct" : ret,
            "n"       : n,
        })

plot_df = pd.DataFrame(plot_rows)

signal_palette = {
    "Score > BIL (above cash)": C_ABOV,    "Score ≤ BIL(below cash)": C_BELO,
}

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for ax, ticker in zip(axes, DEFENSIVE_ASSETS):
    sub_plot = plot_df[plot_df["ticker"] == ticker].copy()
    uncond   = analysis_df[analysis_df["ticker"] == ticker]["fwd_return"].mean() * 100
    color    = COLORS[ticker]

    sns.barplot(
        data=sub_plot,
        x="signal", y="ret_pct",
        palette=signal_palette,
        order=list(signal_palette.keys()),
        edgecolor="white", linewidth=0.8,
        alpha=0.88, ax=ax,
    )

    # Annotate bars with return and observation count
    for patch, (_, row) in zip(ax.patches, sub_plot.iterrows()):
        val  = row["ret_pct"]
        n    = row["n"]
        yoff = 0.06 if val >= 0 else -0.10
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            val + yoff,
            f"{val:+.2f}%n={n}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9, fontweight="bold",
            color="#222222",
        )

    # Zero baseline and unconditional average
    ax.axhline(0,      color="black", linewidth=0.8, alpha=0.5)
    ax.axhline(uncond, color=color,   linewidth=1.8,
               linestyle="--", alpha=0.75,
               label=f"Unconditional avg: {uncond:+.2f}%")

    # Spread annotation
    vals     = sub_plot["ret_pct"].tolist()
    spread   = vals[0] - vals[1]
    ax.text(0.98, 0.04,
            f"Spread: {spread:+.2f}%",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#cccccc",
                      alpha=0.8))

    ax.set_title(ticker, fontsize=12, fontweight="bold", pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("Avg fwd 1-month return (%)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%")
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.tick_params(axis="x", labelsize=9)

period = (f"{monthly.index[0].strftime('%b %Y')} – "
          f"{monthly.index[-1].strftime('%b %Y')}")
fig.suptitle(
    f"BIL Cash Filter — Avg Forward 1-Month Return: Above vs Below Cash\n{period}",
    fontsize=11, fontweight="bold", y=1.01,
)

legend_elements = [
    Patch(facecolor=C_ABOV, label="Score > BIL — momentum exceeds cash"),
    Patch(facecolor=C_BELO, label="Score ≤ BIL — momentum below cash (→ SPY in strategy)"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.04),
           framealpha=0.9)

plt.tight_layout()
_path = os.path.join(OUTPUT_DIR, "df_momentum_predictive_power.png")
plt.savefig(_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"§13 Chart saved: {_path}")


# ══════════════════════════════════════════════════════════════════════════
# §14 COMBINED CHART — ALL ETFs SIDE BY SIDE (seaborn grouped bar)
# ══════════════════════════════════════════════════════════════════════════
# %%

# Long-form data: above BIL vs below BIL for each ticker
combined_rows = []
for ticker in DEFENSIVE_ASSETS:
    sub = analysis_df[analysis_df["ticker"] == ticker]
    for label, mask in [("Above BIL", sub["pos_vs_bil"]),
                         ("Below BIL", ~sub["pos_vs_bil"])]:
        combined_rows.append({
            "ETF"    : ticker,
            "Signal" : label,
            "Return" : sub.loc[mask, "fwd_return"].mean() * 100,
            "N"      : int(mask.sum()),
        })

combined_df = pd.DataFrame(combined_rows)

fig, ax = plt.subplots(figsize=(11, 5.5))

sns.barplot(
    data=combined_df,
    x="ETF", y="Return",
    hue="Signal",
    palette={"Above BIL": C_ABOV, "Below BIL": C_BELO},
    edgecolor="white", linewidth=0.8,
    alpha=0.88, ax=ax,
)

# Annotate bars
for patch, (_, row) in zip(ax.patches, combined_df.iterrows()):
    val = row["Return"]
    n   = row["N"]
    yoff = 0.03 if val >= 0 else -0.06
    ax.text(
        patch.get_x() + patch.get_width() / 2,
        val + yoff,
        f"{val:+.2f}%n={n}",
        ha="center",
        va="bottom" if val >= 0 else "top",
        fontsize=8.5, fontweight="bold", color="#222222",
    )

# Unconditional average per ETF as a dot marker
for i, ticker in enumerate(DEFENSIVE_ASSETS):
    uncond = analysis_df[analysis_df["ticker"] == ticker]["fwd_return"].mean() * 100
    ax.plot(i, uncond, marker="D", markersize=7,
            color=COLORS[ticker], zorder=5,
            label=f"{ticker} unconditional avg" if i == 0 else "")
    ax.text(i + 0.25, uncond, f" {uncond:+.2f}%",
            va="center", fontsize=7.5, color=COLORS[ticker], fontweight="bold")

ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax.set_xlabel("")
ax.set_ylabel("Avg forward 1-month return (%)")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%")
)
ax.set_title(
    "Momentum Signal vs Cash Filter — Avg Forward 1-Month Return by ETF\n"
    f"({monthly.index[0].strftime('%b %Y')} – "
    f"{monthly.index[-1].strftime('%b %Y')})",
    fontsize=11, fontweight="bold", pad=8,
)

# Legend: seaborn hue + unconditional marker
handles, labels = ax.get_legend_handles_labels()
uncond_marker = plt.Line2D([0], [0], marker="D", color="gray",
                            markersize=7, linestyle="None",
                            label="Unconditional avg (per ETF)")
ax.legend(handles=handles + [uncond_marker],
          fontsize=8.5, loc="upper right", framealpha=0.9)

sns.despine(left=False, bottom=False)
plt.tight_layout()
_path2 = os.path.join(OUTPUT_DIR, "df_momentum_combined.png")
plt.savefig(_path2, dpi=150, bbox_inches="tight")
plt.show()
print(f"§14 Chart saved: {_path2}")




# ══════════════════════════════════════════════════════════════════════════
# §14b QUARTILE CHART — return by score quartile (seaborn)
# ══════════════════════════════════════════════════════════════════════════
# %%

quart_rows = []
for ticker in DEFENSIVE_ASSETS:
    sub = analysis_df[analysis_df["ticker"] == ticker].copy()
    sub["score_quartile"] = pd.qcut(
        sub["score"], q=4,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )
    for q, grp in sub.groupby("score_quartile", observed=True):
        quart_rows.append({
            "ETF"      : ticker,
            "Quartile" : str(q),
            "Return"   : grp["fwd_return"].mean() * 100,
            "N"        : len(grp),
        })

quart_df = pd.DataFrame(quart_rows)

fig, ax = plt.subplots(figsize=(12, 5))

sns.barplot(
    data=quart_df,
    x="ETF", y="Return",
    hue="Quartile",
    palette=sns.color_palette("RdYlGn", 4),
    edgecolor="white", linewidth=0.8,
    alpha=0.88, ax=ax,
)

# Annotate
for patch, (_, row) in zip(ax.patches, quart_df.iterrows()):
    val = row["Return"]
    if abs(val) > 0.01:
        yoff = 0.025 if val >= 0 else -0.05
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            val + yoff,
            f"{val:+.2f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=7, fontweight="bold", color="#222222",
        )

ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax.set_xlabel("")
ax.set_ylabel("Avg forward 1-month return (%)")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%")
)
ax.set_title(
    "Average Forward Return by Momentum Score Quartile\n"
    "Q1 = weakest momentum, Q4 = strongest momentum",
    fontsize=11, fontweight="bold", pad=8,
)
ax.legend(title="Score quartile", fontsize=8, loc="upper left",
          framealpha=0.9)

sns.despine()
plt.tight_layout()
_path3 = os.path.join(OUTPUT_DIR, "df_momentum_quartiles.png")
plt.savefig(_path3, dpi=150, bbox_inches="tight")
plt.show()
print(f"§14b Chart saved: {_path3}")

# ══════════════════════════════════════════════════════════════════════════
# §15 HIT RATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
# %%

print(f"\n{'═'*65}")
print("  HIT RATE ANALYSIS")
print("  % of months where positive signal → positive forward return")
print(f"{'═'*65}")
print(f"\n  {'Ticker':>6}  {'Hit Rate (vs 0)':>16}  "
      f"{'Hit Rate (vs BIL)':>18}  {'Base Rate':>10}")
print(f"  {'─'*6}  {'─'*16}  {'─'*18}  {'─'*10}")

hit_rows = []
for ticker in DEFENSIVE_ASSETS:
    sub  = analysis_df[analysis_df["ticker"] == ticker]

    # vs zero: when score>0, what % had positive fwd return?
    pos_zero = sub[sub["pos_vs_zero"]]
    hit_zero = (pos_zero["fwd_return"] > 0).mean()

    # vs BIL: when score>BIL, what % had positive fwd return?
    pos_bil  = sub[sub["pos_vs_bil"]]
    hit_bil  = (pos_bil["fwd_return"] > 0).mean()

    # Base rate: unconditional % of months with positive return
    base     = (sub["fwd_return"] > 0).mean()

    print(f"  {ticker:>6}  {hit_zero:>16.1%}  {hit_bil:>18.1%}  {base:>10.1%}")
    hit_rows.append({
        "Ticker"          : ticker,
        "Hit Rate vs 0"   : hit_zero,
        "Hit Rate vs BIL" : hit_bil,
        "Base Rate"       : base,
    })

hit_df = pd.DataFrame(hit_rows).set_index("Ticker")

# ── Quartile analysis — does higher score = higher return? ────────────────
print(f"\n{'═'*65}")
print("  QUARTILE ANALYSIS — avg forward return by score quartile")
print(f"{'═'*65}")

for ticker in DEFENSIVE_ASSETS:
    sub = analysis_df[analysis_df["ticker"] == ticker].copy()
    q_means = sub.groupby("score_quartile", observed=True)["fwd_return"].agg(
        ["mean", "count"]
    )
    print(f"\n  {ticker}")
    print(f"  {'Quartile':>14}  {'Avg Fwd Return':>15}  {'N':>5}")
    print(f"  {'─'*14}  {'─'*15}  {'─'*5}")
    for q, row in q_means.iterrows():
        print(f"  {str(q):>14}  {row['mean']:>+15.3%}  {row['count']:>5.0f}")


# ══════════════════════════════════════════════════════════════════════════
# §16 SAVE MOMENTUM RESULTS
# ══════════════════════════════════════════════════════════════════════════
# %%

_summary_path  = os.path.join(OUTPUT_DIR, "df_momentum_summary.csv")
_analysis_path = os.path.join(OUTPUT_DIR, "df_momentum_analysis.csv")
_hit_path      = os.path.join(OUTPUT_DIR, "df_momentum_hit_rates.csv")
_quart_path    = os.path.join(OUTPUT_DIR, "df_momentum_quartiles.csv")

mom_summary_df.to_csv(_summary_path)
analysis_df.to_csv(_analysis_path, index=False)
hit_df.to_csv(_hit_path)
quart_df.to_csv(_quart_path, index=False)

print(f"\n§16 Results saved:")
print(f"    {_summary_path}")
print(f"    {_analysis_path}")
print(f"    {_hit_path}    {_quart_path}")
print(f"\nVariables in Explorer:")
print(f"  analysis_df  — row per (month × ticker): score, signal states, fwd return")
print(f"  mom_summary_df   — avg returns by signal state per ticker")
print(f"  hit_df       — hit rates and base rates per ticker  quart_df     — returns by score quartile per ticker")