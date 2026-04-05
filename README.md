# DefenseFirst

Defense First: A Multi-Asset Tactical Allocation Backtest
=========================================================
Paper: Thomas D. Carlson, July 1, 2025

Strategy: Rotates monthly among 4 defensive assets (TLT, GLD, DBC, UUP)
using equal-weighted momentum across 1/3/6/12-month lookbacks with a
40/30/20/10 tiered allocation. Absolute momentum filter vs. BIL (cash)
redirects weak slots to SPY as equity fallback.
