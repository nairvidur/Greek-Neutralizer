# Phase 2 evidence: reconciliation

Gate: "reconciled with QuantLib and with a real day's OPRA-derived chain."

**Status: met, with the limitations below.**

## QuantLib

Run with `uv run pytest -m reference`.

| Item | Reference | Tolerance |
|---|---|---|
| European price, delta, gamma, vega, theta, rho | `AnalyticEuropeanEngine` | 1e-10 relative |
| Implied vol | `blackFormulaImpliedStdDev` | 1e-10 to truth, 1e-6 to QuantLib |
| American, no dividends | `QdFpAmericanEngine` high precision (ALO) | 0.1 cents per 100 of spot |
| American, discrete dividends | FD engine, `Escrowed` model | 0.25 cents |
| American Greeks | FD engine at 1600 by 1600 | 0.2 cents, 0.002 delta, 0.002 gamma |

QuantLib's default dividend model is `Spot`, which prices the test cases about 0.12 higher than `Escrowed`.
That is a modeling choice, not an error, and this engine documents and uses the escrowed assumption.

## Real chains

Source: Cboe's public delayed-quotes endpoint, retrieved 2026-11-01, with quotes from the 2026-10-30 close.
This is delayed data used once for reconciliation, not a licensed feed.
Quote files are not committed, and only the aggregates in `reconcile_SPX.json` and `reconcile_SPY.json` are.

Reproduce (the discount rate for SPY is the SPX box-implied rate, because parity on American options cannot give it):

```sh
curl -L -H 'User-Agent: Mozilla/5.0' https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json -o vendor_data/_SPX.json
uv run python scripts/reconcile_chain.py vendor_data/_SPX.json
```

Re-running against a later download gives different numbers, because the market moved.

The JSON files hold the complete results.
The headline figures are in this table.

| Measure | SPX | SPY |
|---|---|---|
| Quotes in payload | 27,760 | 11,820 (84 non-standard excluded) |
| Expiries fit | 53 of 53 | 30 of 30 |
| Vega-weighted fit error, median | 0.12 vol pts | 0.18 vol pts |
| Quoted-range butterfly, minimum g (must be at least 0) | +0.110 | +0.022 |
| Quoted-range calendar, worst relative gap | -0.006% | -0.001% |
| Implied vol, median difference from Cboe | 0.10 vol pts | 0.19 vol pts |
| Delta, median difference from Cboe | 0.0038 | 0.0015 |
| Gamma, median relative difference from Cboe | 5.5% | 2.1% |

## What real data found

The synthetic tests passed before any of these were fixed, and each was found only by running the real chain:

1. The endpoint leaves quote sizes empty for SPX, so a no-size check blocked every quote.
   Sizes are now an explicit policy, with absence detected.
2. The payload timestamp is the fetch time. The quotes were a day older, which misstates T badly on short expiries.
   As-of now comes from the last trade times.
3. Regressing the discount factor from American C - P gave implied rates of -6%.
   The discount is now an input, and the box rate is computed only from European chains.
4. The index's last print (7580.07) was stale against option parity (7590.09).
   Spot now comes from the nearest parity forward.
5. 84 rows inside SPY were adjusted contracts (root `B1SP`). They are now excluded and counted.
6. `check_grid` measured variance at the SVI `m` parameter rather than at the money, which marked unquoted strikes as core arbitrage.

Two hypotheses were tested and rejected along the way: calendar-floor error propagation, and an election-driven W-shaped smile.
Neither matched the data.

## Limitations

- **Bid/ask:** a parametric surface fits inside the quoted market for only 35% of SPX and 29% of SPY contracts, with a median error of 1.4 and 1.8 half-spreads.
  Five SVI parameters cannot reach every tight quote.
  Therefore hedge costs and marks must come from live quotes, never from the surface, which is used only for risk and interpolation.
- **Beyond the quotes:** arbitrage is verified only inside quoted strikes, and full-grid checks do show calendar crossings beyond them.
  The book reports `EXTRAPOLATED_VOL` for any contract out there.
- **One snapshot:** this is one day at one instant, the close, and it does not establish stability over time.
- **Vendor comparison:** the vendor's Greeks use its own unpublished conventions, so this is a sanity reconciliation, not a benchmark of truth.
  The gamma gap of 5.5% on SPX is larger than the implied vol gap would explain, and the cause is not yet determined.
- **Delayed source:** this does not exercise live feed handling.
