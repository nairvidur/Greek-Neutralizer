# Phase 6: forward paper pilot on free chains

Status: **registered on 2026-11-10, with no readout yet.**
No chain day it scores existed when this protocol was written.

This is the paper-pilot part of phase 6 that needs no partner.
It does not replace the shadow pilot on a design partner's real book, the independent validation, or the penetration test, which still need people.

## Why a forward pilot

The position-aware optimizer variants were designed on SPY and held out on DIA and XLF, all over the same 2020 to 2026 window, so the held-out test shares SPY's market regime.
Days after 2026-11-10 are out of sample in time for every symbol, SPY included, and nothing here can have been tuned on them.
The free source posts each day's chain on the following day, so the pilot is scored by catching up on published days, not by running live.
It tests the hedge decisions on new markets, not the operations of a live desk: order timing, fills, and a partner's own positions stay untested.

## Protocol

Everything below was fixed before any scored day existed.
A change needs a dated entry in the ledger below, made before the readout it affects, with its reason.

- Start: the first chain day on or after **2026-11-11**.
- Symbols: **DIA**, primary, as in the held-out test; **SPY**, secondary; and **XLF**, reported but not decisive, since 231 of its held-out days were unusable for too few strikes.
- Variants: all 25 registered in `scripts/run_chain_backtest.py`, at the books, costs, fees, quote sizes, dividends, rates, and bootstrap of phase 4.
- Tested: `opt_dgv_bounded`, `opt_dgv_bounded_band`, and `opt_dgv_surface`, each against `delta_daily`, on the short strangle at quoted spreads.
- Gate, for each tested variant: **go** if its variance reduction and its 97.5% ES improvement against daily delta both have 95% block-bootstrap intervals above zero on DIA.
  A go is **confirmed** if the same holds on SPY.
  With three variants, one may pass by chance, and every result is reported.
- Stated with any decision, not gated: the mean P&L difference against daily delta, with its interval, which is the carry the hedge gives up.
- Decision: once, at the first readout on or after **2028-11-11**, about 500 chain days.
  The held-out DIA intervals came from 1,180 periods; at 250 periods they would be about 2.2 times as wide, and the ES lower bounds would fall below zero, so a one-year decision would mostly measure the sample size.
  At about 500 periods they are about 1.5 times as wide, which still leaves the ES test close to the line even if the held-out effect is real.
- Interim readouts: at most once a quarter, descriptive only.
  They report every statistic and no decision, and no result in one stops the pilot early or starts its decision early.
  Until a symbol has 40 periods, its readout has no ES and no intervals, and says so.
- Stopping early, only for a defect: failed or time-limited days, a data problem, or a code fault.
  The defect, its fix, and whether the pilot restarts go in the ledger.
- Code: readouts run the code as registered.
  The registered code is the merge commit of the pull request that adds this file into main, entered in the ledger right after the merge.
  Commits made on that pull request before its merge are part of the registration, not later changes.
  If anything under `src/greeks_neutralizer/` or `scripts/run_chain_backtest.py` that the backtest uses has changed since the registered commit, the readout reports results both at that commit and at the current code, and the decision uses the registered code unless a ledger entry says why not.
  Each readout records `git_commit` and `git_dirty`, the commit it ran on and whether tracked files under `src/` or `scripts/` differed from it.
  A readout from a dirty tree is not a pilot result.
- Data: each readout records `inputs_sha256` over every file it read, because the free sources revise their data.
  The downloaded files are vendor data and stay out of version control; keep them for the pilot's length.

## What a go would and would not allow

A confirmed go makes a variant a candidate default for repeated re-hedging of a short-volatility book, still subject to the rest of phase 6.
It would not show that the hedge pays: on the held-out data each variant gave up $790 to $1,150 a period of the strangle's carry.
A result that is not established says the data do not separate the variant from daily delta, not that it is worse.

## Readouts

Run from a checkout of this repository, with the vendor data outside it:

```sh
curl -L 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO' -o ~/gn_pilot/DGS3MO.csv
for sym in DIA SPY XLF; do
  dir=~/gn_pilot/dolthub_$sym
  uv run python scripts/fetch_dolthub_chains.py $dir --symbol $sym --start 2026-11-11
  uv run python scripts/run_chain_backtest.py $dir/chains/$sym $dir/${sym}_dividends.csv \
    ~/gn_pilot/DGS3MO.csv --symbol $sym --start 2026-11-11 \
    --out docs/evidence/phase6/readouts/$(date +%F)_$sym.json
done
```

The fetch probes every weekday from the start and resumes where it stopped.
A recent day with no chain or no close may only be unposted, so for seven days the fetch leaves it unmarked and probes it again on the next run, rather than recording it as missing for good.
Commit each readout's results file, with an entry below.

| Readout | Date | Symbols | Periods | Code | Result |
|---|---|---|---|---|---|
| None yet | | | | | |

## Ledger

- 2026-11-10: protocol registered.
  Before registering, the runner gained `--start`, a digest of its inputs, and readouts of runs too short for intervals or ES, and the fetch gained the settle window above.
  They were checked on DIA chains from 2026-10-01 to 2026-10-30, which are before the start and are not pilot data.
  The registered commit is entered here after the merge.
