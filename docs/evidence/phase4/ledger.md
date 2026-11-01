# Experiment ledger

Every variant tested, so selection stays visible.
An entry is added whenever a variant or an assumption changes.

## Pre-registered variants (16)

| Book | Spread multiple | Policies |
|---|---|---|
| Short strangle | 1 | unhedged, delta_daily, delta_band, opt_dg, opt_dgv, opt_dgv_cheap, opt_dgv_cvar |
| Short strangle | 3 | unhedged, delta_daily, delta_band, opt_dgv, opt_dgv_cheap |
| Long put overlay | 1 | unhedged, delta_daily, opt_dgv, opt_dgv_cheap |

## Runs

| Run | Code | Outcome | Status |
|---|---|---|---|
| 1 | `e973e3e` | Stopped by hand, because the solver was being changed | Discarded, no results read |
| 2 | `7dffb11` | `opt_dgv_cvar` bought 11.6 billion lots and lost $10 trillion, with 318 failed days. Others completed | **Rejected: exposed a defect.** CVaR could be farmed for profit |
| 3 | `062f33a` | All variants completed, with no failed days | **Current:** `proxy_backtest.json` |

## Changes between runs 2 and 3

These were all fixes to defects, none aimed at improving results, and every variant was rerun:

- CVaR floored at zero, with a hard lot cap.
- Default gamma weight corrected from 0.015 to 0.045.
- Accounting, time, and quote gate fixes from independent review.

Reported variance reduction against delta for `opt_dgv` moved from 93.1% to 92.3%.

## Judgment values, set once and not tuned

| Value | Setting | Source |
|---|---|---|
| Smile skew and curvature | -0.10, 0.02 | Stylized |
| ATM as a share of VIX | 0.90 | Stylized, since the VIX sits above ATM vol |
| Option half-spread | 0.15 + 0.2% of premium, scaled up with VIX | Near the SPX levels seen on 2026-10-30, and otherwise assumed |
| Risk aversion | 1.0, and 0.2 for `cheap` | Judgment |
| No-trade band | 0.5 bp cost, risk aversion 1e-6 | Judgment |

## Promotion rule

Nothing here is promoted to a recommended default.
That needs a rerun on historical chains, then a paper pilot.
