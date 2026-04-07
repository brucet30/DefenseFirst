"""
Defense First — UUP Replacement Candidate Analysis
====================================================
Tests momentum signal quality for ETFs that could replace UUP
in the Defense First strategy.

For each candidate ETF:
  1. Compute composite momentum score (same 1/3/6/12m method)
  2. Compare average forward returns when score > BIL vs score < BIL
  3. Compute hit rate, spread, and Sharpe of the momentum signal
  4. Compare correlation with existing defensive assets (TLT, GLD, DBC)

SPYDER USAGE
  Run cells with Ctrl+Enter or full file with F5.
  Results land in Variable Explorer.
"""

# ══════════════════════════════════════════════════════════════════════════
# §1  IMPORTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════
# %%

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Existing strategy assets ──────────────────────────────────────────────
EXISTING     = ["TLT", "GLD", "DBC"]   # what stays
CASH_PROXY   = "BIL"
BENCHMARK    = "SPY"
LOOKBACK_MONTHS = [1, 3, 6, 12]

# ── UUP and replacement candidates ───────────────────────────────────────
UUP          = "UUP"
CANDIDATES   = {
    "UUP" : "US Dollar Index (current)",
    "SHY" : "1-3yr Treasuries",
    "TIP" : "TIPS (Inflation-linked)",
    "BTAL": "Long low-beta / Short high-beta",
    "VIXM": "Mid-term VIX Futures",
    "GVI" : "iShares Govt/Credit Bond",
    "DFNL": "Defensive (placeholder)",  # removed if no data
}

# Ordered list — UUP first for direct comparison
TICKERS = ["TLT", "GLD", "DBC", "UUP", "SHY", "TIP", "BTAL", "VIXM", "BIL", "SPY"]

START_DATE  = "2007-06-01"
END_DATE    = "2026-03-31"
OUTPUT_DIR  = os.path.expanduser("~/Documents")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "UUP" : "#8e44ad",
    "SHY" : "#2471a3",
    "TIP" : "#e67e22",
    "BTAL": "#1a9e75",
    "VIXM": "#c0392b",
    "GVI" : "#7f8c8d",
    "TLT" : "#2980b9",
    "GLD" : "#f39c12",
    "DBC" : "#27ae60",
    "BIL" : "#aaaaaa",
    "SPY" : "#e74c3c",
}

C_ABOV = "#1D9E75"
C_BELO = "#E24B4A"

sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams.update({
    "figure.dpi"      : 120,
    "figure.facecolor": "white",
    "axes.titlesize"  : 10,
    "axes.titleweight": "bold",
})

print("§1  Config loaded.")
print(f"    Candidates: {list(CANDIDATES.keys())}")


# ══════════════════════════════════════════════════════════════════════════
# §2  DOWNLOAD DATA
# ══════════════════════════════════════════════════════════════════════════
# %%

print(f"\nDownloading {TICKERS} ...")
_dl = yf.download(TICKERS, start=START_DATE, end=END_DATE,
                  auto_adjust=True, progress=False)

if isinstance(_dl.columns, pd.MultiIndex):
    _close = [c for c in _dl.columns.get_level_values(0).unique()
              if "close" in str(c).lower()]
    _raw = _dl[_close[0]]
    if isinstance(_raw.columns, pd.MultiIndex):
        _raw.columns = _raw.columns.droplevel(0)
else:
    _raw = _dl["Close"] if "Close" in _dl.columns else _dl

daily = _raw.ffill()

try:
    monthly = daily.resample("ME").last()
except ValueError:
    monthly = daily.resample("M").last()

# Remove tickers with insufficient data (< 36 months)
valid = [t for t in TICKERS if t in monthly.columns
         and monthly[t].dropna().shape[0] >= 36]
missing = [t for t in TICKERS if t not in valid]
if missing:
    print(f"  Dropped (insufficient data): {missing}")

monthly = monthly[valid]

print(f"  {len(monthly)} monthly bars  "
      f"({monthly.index[0].strftime('%Y-%m')} → "
      f"{monthly.index[-1].strftime('%Y-%m')})")
print(f"  Valid tickers: {valid}")

# Per-ticker inception dates
print(f"\n  {'Ticker':>6}  {'First bar':>10}  {'Months':>7}")
print(f"  {'─'*6}  {'─'*10}  {'─'*7}")
for t in valid:
    s = monthly[t].dropna()
    print(f"  {t:>6}  {s.index[0].strftime('%Y-%m'):>10}  {len(s):>7}")


# ══════════════════════════════════════════════════════════════════════════
# §3  COMPUTE MOMENTUM SCORES & FORWARD RETURNS
# ══════════════════════════════════════════════════════════════════════════
# %%

def composite_score(series):
    """Equal-weighted avg of 1/3/6/12-month simple returns."""
    parts = [series.pct_change(lb) for lb in LOOKBACK_MONTHS]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False)

# Compute scores for all tickers
scores = pd.DataFrame({t: composite_score(monthly[t])
                        for t in valid if t in monthly.columns})

# Forward 1-month returns
fwd_rets = monthly.pct_change().shift(-1)

# Cash benchmark score
cash_score = scores[CASH_PROXY] if CASH_PROXY in scores.columns else None

# Analysis candidates — all except BIL, SPY, existing (TLT, GLD, DBC)
# Include UUP for direct comparison
analyse = [t for t in valid if t not in [CASH_PROXY, BENCHMARK]
           and t in scores.columns]

print(f"§3  Momentum scores computed for: {analyse}")


# ══════════════════════════════════════════════════════════════════════════
# §4  SIGNAL QUALITY METRICS PER CANDIDATE
# ══════════════════════════════════════════════════════════════════════════
# %%

results = []

for ticker in analyse:
    if ticker not in scores.columns or ticker not in fwd_rets.columns:
        continue

    # Align on valid index (both score and forward return available)
    df = pd.DataFrame({
        "score"    : scores[ticker],
        "cash"     : cash_score,
        "fwd_ret"  : fwd_rets[ticker],
    }).dropna()

    if len(df) < 24:
        print(f"  {ticker}: skipped (only {len(df)} valid obs)")
        continue

    above = df[df["score"] >= df["cash"]]
    below = df[df["score"] <  df["cash"]]

    ret_above   = above["fwd_ret"].mean()
    ret_below   = below["fwd_ret"].mean()
    ret_all     = df["fwd_ret"].mean()
    spread      = ret_above - ret_below
    hit_above   = (above["fwd_ret"] > 0).mean() if len(above) > 0 else np.nan
    n_above     = len(above)
    n_below     = len(below)
    pct_above   = n_above / len(df)

    # Sharpe of a simple long-when-above strategy
    signal_rets = pd.Series(np.where(
        df["score"] >= df["cash"], df["fwd_ret"], 0
    ), index=df.index)
    sig_sharpe  = (signal_rets.mean() / signal_rets.std() * np.sqrt(12)
                   if signal_rets.std() > 0 else 0)

    # Unconditional stats
    uncond_cagr = (1 + df["fwd_ret"]).prod() ** (12 / len(df)) - 1
    uncond_vol  = df["fwd_ret"].std() * np.sqrt(12)

    # Correlation with existing defensive assets
    corr_tlt = monthly["TLT"].pct_change().corr(monthly[ticker].pct_change()) \
               if "TLT" in monthly.columns else np.nan
    corr_gld = monthly["GLD"].pct_change().corr(monthly[ticker].pct_change()) \
               if "GLD" in monthly.columns else np.nan
    corr_dbc = monthly["DBC"].pct_change().corr(monthly[ticker].pct_change()) \
               if "DBC" in monthly.columns else np.nan
    corr_spy = monthly["SPY"].pct_change().corr(monthly[ticker].pct_change()) \
               if "SPY" in monthly.columns else np.nan

    results.append({
        "Ticker"       : ticker,
        "Label"        : CANDIDATES.get(ticker, ticker),
        "N_obs"        : len(df),
        "N_above"      : n_above,
        "N_below"      : n_below,
        "Pct_above"    : pct_above,
        "Ret_above"    : ret_above,
        "Ret_below"    : ret_below,
        "Ret_all"      : ret_all,
        "Spread"       : spread,
        "Hit_above"    : hit_above,
        "Sig_Sharpe"   : sig_sharpe,
        "CAGR"         : uncond_cagr,
        "Vol"          : uncond_vol,
        "Corr_TLT"     : corr_tlt,
        "Corr_GLD"     : corr_gld,
        "Corr_DBC"     : corr_dbc,
        "Corr_SPY"     : corr_spy,
    })

res_df = pd.DataFrame(results).set_index("Ticker")

# ── Print results table ───────────────────────────────────────────────────
print(f"\n{'═'*90}")
print("  MOMENTUM SIGNAL QUALITY — CANDIDATES vs UUP")
print(f"  Signal: composite momentum score > BIL cash score")
print(f"{'═'*90}")

print(f"\n  {'Ticker':>6}  {'N':>5}  {'%>BIL':>6}  "
      f"{'Ret(>BIL)':>10}  {'Ret(<BIL)':>10}  "
      f"{'Spread':>8}  {'Hit%':>6}  {'SigSharpe':>10}  {'CAGR':>7}")
print(f"  {'─'*6}  {'─'*5}  {'─'*6}  "
      f"{'─'*10}  {'─'*10}  "
      f"{'─'*8}  {'─'*6}  {'─'*10}  {'─'*7}")

for t, row in res_df.iterrows():
    flag = "  ← current" if t == "UUP" else ""
    print(f"  {t:>6}  "
          f"{row['N_obs']:>5.0f}  "
          f"{row['Pct_above']:>6.1%}  "
          f"{row['Ret_above']:>+10.3%}  "
          f"{row['Ret_below']:>+10.3%}  "
          f"{row['Spread']:>+8.3%}  "
          f"{row['Hit_above']:>6.1%}  "
          f"{row['Sig_Sharpe']:>10.2f}  "
          f"{row['CAGR']:>+7.2%}"
          f"{flag}")

# ── Correlation table ─────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print("  CORRELATION WITH EXISTING DEFENSIVE ASSETS")
print(f"{'═'*60}")
print(f"\n  {'Ticker':>6}  {'vs TLT':>8}  {'vs GLD':>8}  "
      f"{'vs DBC':>8}  {'vs SPY':>8}")
print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
for t, row in res_df.iterrows():
    print(f"  {t:>6}  "
          f"{row['Corr_TLT']:>8.2f}  "
          f"{row['Corr_GLD']:>8.2f}  "
          f"{row['Corr_DBC']:>8.2f}  "
          f"{row['Corr_SPY']:>8.2f}")


# ══════════════════════════════════════════════════════════════════════════
# §5  CHART 1 — SIGNAL QUALITY COMPARISON (grouped bars)
# ══════════════════════════════════════════════════════════════════════════
# %%

# Build long-form for seaborn
plot_rows = []
for t, row in res_df.iterrows():
    for label, val in [("Above BIL", row["Ret_above"]),
                        ("Below BIL", row["Ret_below"])]:
        plot_rows.append({
            "Ticker" : t,
            "Signal" : label,
            "Return" : val * 100,
        })
plot_df = pd.DataFrame(plot_rows)

bar_colors = [COLORS.get(t, "#aaaaaa") for t in res_df.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left: Above vs Below BIL returns ─────────────────────────────────────
ax = axes[0]
sns.barplot(
    data=plot_df,
    x="Ticker", y="Return",
    hue="Signal",
    palette={"Above BIL": C_ABOV, "Below BIL": C_BELO},
    edgecolor="white", linewidth=0.8,
    alpha=0.88, ax=ax,
)

for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
    val  = row["Return"]
    yoff = 0.03 if val >= 0 else -0.06
    ax.text(
        patch.get_x() + patch.get_width() / 2,
        val + yoff,
        f"{val:+.2f}%",
        ha="center",
        va="bottom" if val >= 0 else "top",
        fontsize=7.5, fontweight="bold", color="#222222",
    )

# Highlight UUP bar group
uup_idx = list(res_df.index).index("UUP")
ax.axvspan(uup_idx - 0.4, uup_idx + 0.4, alpha=0.08,
           color="purple", label="Current (UUP)")

ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax.set_title("Avg Forward 1-Month Return\nAbove vs Below BIL Momentum Filter")
ax.set_xlabel("")
ax.set_ylabel("Avg forward 1-month return (%)")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%")
)
ax.legend(fontsize=8)

# ── Right: Spread and Signal Sharpe ──────────────────────────────────────
ax2 = axes[1]
x    = np.arange(len(res_df))
w    = 0.35

spreads  = res_df["Spread"].values * 100
sharpes  = res_df["Sig_Sharpe"].values
clrs     = [COLORS.get(t, "#aaaaaa") for t in res_df.index]

bars1 = ax2.bar(x - w/2, spreads, w,
                color=clrs, alpha=0.85,
                edgecolor="white", linewidth=0.8,
                label="Spread (above − below) %")

ax2b = ax2.twinx()
bars2 = ax2b.bar(x + w/2, sharpes, w,
                 color=clrs, alpha=0.45,
                 edgecolor="white", linewidth=0.8,
                 hatch="///", label="Signal Sharpe")

# Annotations
for bar, val in zip(bars1, spreads):
    yoff = 0.02 if val >= 0 else -0.05
    ax2.text(bar.get_x() + bar.get_width()/2, val + yoff,
             f"{val:+.2f}%", ha="center",
             va="bottom" if val >= 0 else "top",
             fontsize=7.5, fontweight="bold")

for bar, val in zip(bars2, sharpes):
    yoff = 0.02 if val >= 0 else -0.04
    ax2b.text(bar.get_x() + bar.get_width()/2, val + yoff,
              f"{val:.2f}", ha="center",
              va="bottom" if val >= 0 else "top",
              fontsize=7.5, color="#555555")

ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax2.axvspan(uup_idx - 0.45, uup_idx + 0.45, alpha=0.08, color="purple")
ax2.set_xticks(x)
ax2.set_xticklabels(res_df.index, fontsize=9)
ax2.set_ylabel("Spread (%)", color="#333333")
ax2b.set_ylabel("Signal Sharpe", color="#777777")
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%")
)
ax2.set_title("Momentum Spread & Signal Sharpe\n(higher is better for both)")

# Combined legend
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

period = (f"{monthly.index[0].strftime('%b %Y')} – "
          f"{monthly.index[-1].strftime('%b %Y')}")
fig.suptitle(
    f"UUP Replacement Candidates — Momentum Signal Quality  |  {period}",
    fontsize=11, fontweight="bold", y=1.01,
)
plt.tight_layout()
_path1 = os.path.join(OUTPUT_DIR, "df_uup_candidates_signal.png")
plt.savefig(_path1, dpi=150, bbox_inches="tight")
plt.show()
print(f"§5  Chart saved: {_path1}")


# ══════════════════════════════════════════════════════════════════════════
# §6  CHART 2 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════
# %%

# Compute full correlation matrix of monthly returns
all_candidates = [t for t in valid if t != BENCHMARK]
ret_matrix = monthly[all_candidates].pct_change().dropna()
corr_matrix = ret_matrix.corr()

fig, ax = plt.subplots(figsize=(9, 7))

mask = np.zeros_like(corr_matrix, dtype=bool)
# No mask — show full matrix

sns.heatmap(
    corr_matrix,
    annot=True, fmt=".2f",
    cmap="RdYlGn",
    center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 8},
    ax=ax,
)

ax.set_title(
    "Monthly Return Correlations — All Candidates\n"
    "(lower correlation with TLT/GLD/DBC = better diversification)",
    fontsize=10, fontweight="bold", pad=10,
)

# Highlight the candidate rows/cols
candidate_tickers = [t for t in ["UUP","SHY","TIP","BTAL","VIXM"]
                     if t in corr_matrix.index]
labels = list(corr_matrix.index)
for t in candidate_tickers:
    if t in labels:
        idx = labels.index(t)
        ax.add_patch(plt.Rectangle((idx, 0), 1, len(labels),
                                    fill=False, edgecolor="#333333",
                                    linewidth=1.5, clip_on=False))
        ax.add_patch(plt.Rectangle((0, idx), len(labels), 1,
                                    fill=False, edgecolor="#333333",
                                    linewidth=1.5, clip_on=False))

plt.tight_layout()
_path2 = os.path.join(OUTPUT_DIR, "df_uup_candidates_corr.png")
plt.savefig(_path2, dpi=150, bbox_inches="tight")
plt.show()
print(f"§6  Chart saved: {_path2}")


# ══════════════════════════════════════════════════════════════════════════
# §7  CHART 3 — CUMULATIVE RETURN COMPARISON
# ══════════════════════════════════════════════════════════════════════════
# %%

fig, ax = plt.subplots(figsize=(12, 5))

plot_tickers = [t for t in ["UUP","SHY","TIP","BTAL","VIXM"]
                if t in monthly.columns]

for t in plot_tickers:
    s = monthly[t].dropna()
    cum = (s / s.iloc[0] * 100)
    ls  = "--" if t == "UUP" else "-"
    lw  = 1.2 if t == "UUP" else 1.6
    ax.plot(cum.index, cum, label=t, color=COLORS.get(t, "#aaaaaa"),
            linestyle=ls, linewidth=lw, alpha=0.9)
    ax.annotate(
        f"{t} ({cum.iloc[-1]:.0f})",
        xy=(cum.index[-1], cum.iloc[-1]),
        xytext=(6, 0), textcoords="offset points",
        fontsize=7.5, color=COLORS.get(t, "#aaaaaa"), va="center",
    )

ax.axhline(100, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
ax.set_title(
    "Cumulative Total Return — UUP vs Candidates (rebased=100)\n"
    f"Start date varies by ETF inception",
    fontsize=10, fontweight="bold", pad=8,
)
ax.set_ylabel("Index (rebased = 100)")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.0f}")
)
ax.legend(fontsize=8, loc="upper left")
sns.despine()
plt.tight_layout()
_path3 = os.path.join(OUTPUT_DIR, "df_uup_candidates_returns.png")
plt.savefig(_path3, dpi=150, bbox_inches="tight")
plt.show()
print(f"§7  Chart saved: {_path3}")


# ══════════════════════════════════════════════════════════════════════════
# §8  SUMMARY SCORECARD
# ══════════════════════════════════════════════════════════════════════════
# %%

print(f"\n{'═'*75}")
print("  REPLACEMENT SCORECARD")
print("  Scoring: Spread (+), Signal Sharpe (+), Low corr with TLT/GLD/DBC (+)")
print(f"{'═'*75}")

# Simple composite score:
# Normalise spread, signal sharpe, and avg abs correlation to [0,1]
# Higher spread = better, higher sharpe = better, lower avg corr = better

cands = [t for t in res_df.index if t in ["UUP","SHY","TIP","BTAL","VIXM"]]
sc    = res_df.loc[cands].copy()

sc["avg_corr_exist"] = sc[["Corr_TLT","Corr_GLD","Corr_DBC"]].abs().mean(axis=1)

def norm(s, higher_better=True):
    rng = s.max() - s.min()
    if rng == 0: return pd.Series(0.5, index=s.index)
    n = (s - s.min()) / rng
    return n if higher_better else 1 - n

sc["score_spread"]  = norm(sc["Spread"])
sc["score_sharpe"]  = norm(sc["Sig_Sharpe"])
sc["score_decorr"]  = norm(sc["avg_corr_exist"], higher_better=False)
sc["score_cagr"]    = norm(sc["CAGR"])
sc["composite"]     = (sc["score_spread"] * 0.35 +
                        sc["score_sharpe"] * 0.35 +
                        sc["score_decorr"] * 0.20 +
                        sc["score_cagr"]   * 0.10)

sc_sorted = sc.sort_values("composite", ascending=False)

print(f"\n  {'Ticker':>6}  {'Spread':>8}  {'SigSharpe':>10}  "
      f"{'AvgCorr':>8}  {'CAGR':>7}  {'Composite':>10}  Rank")
print(f"  {'─'*6}  {'─'*8}  {'─'*10}  "
      f"{'─'*8}  {'─'*7}  {'─'*10}  {'─'*4}")

for rank, (t, row) in enumerate(sc_sorted.iterrows(), 1):
    flag = "  ← CURRENT" if t == "UUP" else ""
    print(f"  {t:>6}  "
          f"{row['Spread']*100:>+8.3f}%  "
          f"{row['Sig_Sharpe']:>10.2f}  "
          f"{row['avg_corr_exist']:>8.2f}  "
          f"{row['CAGR']*100:>+7.2f}%  "
          f"{row['composite']:>10.2f}  "
          f"#{rank}{flag}")

winner = sc_sorted.index[0]
print(f"\n  → Best replacement candidate: {winner} "
      f"({CANDIDATES.get(winner, winner)})")
print(f"     Spread: {sc_sorted.loc[winner,'Spread']*100:+.3f}%  "
      f"Signal Sharpe: {sc_sorted.loc[winner,'Sig_Sharpe']:.2f}  "
      f"Avg corr vs existing: {sc_sorted.loc[winner,'avg_corr_exist']:.2f}")


# ══════════════════════════════════════════════════════════════════════════
# §9  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════
# %%

_res_path   = os.path.join(OUTPUT_DIR, "df_uup_candidates.csv")
_score_path = os.path.join(OUTPUT_DIR, "df_uup_scorecard.csv")

res_df.to_csv(_res_path)
sc_sorted[["Spread","Sig_Sharpe","avg_corr_exist","CAGR","composite"]].to_csv(_score_path)

print(f"\n§9  Results saved:")
print(f"    {_res_path}")
print(f"    {_score_path}")
print(f"\nVariables in Explorer:")
print(f"  res_df     — signal quality metrics per candidate")
print(f"  sc_sorted  — composite scorecard, ranked")
print(f"  corr_matrix — full return correlation matrix")
