# Model governance pack

A starting point for a firm's own model risk process, shaped around SR 26-2 (April 2026, replacing SR 11-7), OCC Bulletin 2026-13, and PRA SS1/23.
Check the primary texts: these references rest on secondary summaries, and whether any of this applies depends on your firm.

**No independent validation has been performed.**
Everything below was produced by the authors, so it is an input to validation and not a substitute for it.

## Inventory

| Model | Purpose | Output used for | Module |
|---|---|---|---|
| Forward Black-Scholes | European value and Greeks | Risk, hedge exposures | `pricing/black.py` |
| Leisen-Reimer tree | American value and Greeks | Risk, hedge exposures | `pricing/american.py` |
| Implied vol solver | Quote to vol | Surface inputs | `pricing/implied_vol.py` |
| SVI surface | Vol by strike and tenor | Risk and interpolation, **not marks or costs** | `surface.py` |
| Parity forward | Forward per expiry | Moneyness | `data/chain.py` |
| Hedge optimizer | Trade recommendation | Proposals, subject to human approval | `hedging/` |
| Scenario engine | Stress P&L | Risk reporting, CVaR | `risk/scenarios.py` |
| VaR and ES | Loss distribution | Risk reporting | `risk/measures.py` |
| Margin estimates | Rough capital use | Comparison only, **never a margin call** | `risk/margin.py` |

## Assumptions that matter

| Assumption | If wrong | Monitor by |
|---|---|---|
| Lognormal per contract at its implied vol | Greeks misstate risk for large moves | `taylor_vs_full`, stress by repricing |
| Escrowed dividends | American prices shift: about 0.12 in the tested calls | Reconcile with your desk's convention |
| Smile regime is sticky strike in scenarios | Skew-related P&L is misestimated | Compare `sticky_strike=False` |
| ACT/365 clock by default | Short-dated Greeks misstate time decay | Use a `VarianceClock`, consistently with the vols |
| Spreads and impact as supplied | Hedge costs are misstated | Post-trade slippage against `expected_price` |
| Hedge weights are desk judgment | The trade-off reflects someone's opinion | Review weights with the desk |

## Testing performed

See `docs/evidence/`.

| Test | Result |
|---|---|
| Analytic Greeks against finite differences | Pass |
| European against QuantLib | Within 1e-10 relative |
| American against QuantLib | Within 0.1 cents per 100 of spot; 0.25 cents with dividends |
| Real SPX and SPY chain against Cboe's reported values | IV within 0.10 and 0.19 vol pts, median |
| Hedged book against QuantLib | Greeks within about 1e-11 |
| Backtest on real paths with modeled option prices | See `docs/evidence/phase4/` |
| Backtest on free historical SPY chains | No-go for the optimizer as first configured, against daily delta; see `docs/evidence/phase4/` |
| Position-aware optimizer, held out on DIA and XLF | Go on DIA, not confirmed on XLF; eligible for a paper pilot, not a default |

## Testing NOT performed

- Backtesting on a licensed chain with every listed strike. The free chain used for phase 4 rarely quotes the same contract twice, so most marks there came from the model.
- Reconciliation to broker or clearing statements.
- Any live pilot, or a paper pilot with a readout: the forward paper pilot in `docs/evidence/phase6/` was registered on 2026-11-10 and decides in 2028.
- Independent validation.
- Long-run monitoring of surface fit, or of hedge effectiveness.

## Ongoing monitoring a user should run

- Daily unexplained P&L from `attribute_pnl`, tracked against a threshold.
- VaR exceptions through `backtest_var`.
- Surface fit error, and how often `arbitrage_quoted_clean` fails.
- Share of proposals that are `unstable`, or `within_gap`.
- Realized cost against proposed cost.

## Change control

- Versions are semantic. The code version is in every manifest.
- A golden or evidence change needs an explanation in its PR.
- Variants tested in backtests are logged in `docs/evidence/phase4/ledger.md`.

## Validator reproduction

```sh
uv sync
uv run pytest -q                 # everything
uv run pytest -m reference -q    # QuantLib reconciliations
```

Scripts in `scripts/` regenerate the evidence.
Vendor data is not committed, and `docs/evidence/phase2/README.md` shows how to fetch it.
Because market data changes, a later download gives different numbers from those committed.
