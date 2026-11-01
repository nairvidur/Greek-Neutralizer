# Stress scenarios

Defined in `risk/scenarios.py`.
Each one reprices in full.

## Provenance

Compare the table with FRED by running:

```sh
curl -L 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS' -o VIXCLS.csv
curl -L 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500' -o SP500.csv
uv run python scripts/check_stress_table.py VIXCLS.csv SP500.csv
```

| Scenario | Spot | VIX change | Spot source | Vol source |
|---|---|---|---|---|
| 1987-10-19 | -20.47% | +60 pts | Not checked here | **Judgment**: the VIX did not exist |
| 2008-09-29 | -8.79% | +11.98 | Not checked here | FRED VIXCLS |
| 2008-10-15 | -9.03% | +14.12 | Not checked here | FRED VIXCLS |
| 2015-08-24 | -3.94% | +12.71 | Not checked here | FRED VIXCLS |
| 2018-02-05 | -4.10% | +20.01 | FRED SP500 | FRED VIXCLS |
| 2020-03-16 | -11.98% | +24.86 | FRED SP500 | FRED VIXCLS |
| 2024-08-05 | -3.00% | +15.18 | FRED SP500 | FRED VIXCLS |

FRED's SP500 series covers only the last ten years, so the earlier spot moves come from general knowledge and should be checked against a licensed source.

## Assumptions

- Skew shocks are **judgment values** for all scenarios, since neither the VIX nor the index gives skew.
- The VIX change is applied as the 30-day ATM vol change, though the VIX is not ATM vol.
- Vol shocks scale with tenor as `(30d / tau)^0.5` by default.
- `sticky_strike` defaults to true: vol at a fixed strike only changes through the stated shocks.
- All are one-day moves applied instantly, with no rebalancing, and liquidity is not stressed.

## Using them

A scenario library is not a forecast.
A book can pass all of these and still fail on a move no one listed.
Add firm-specific scenarios through a request's `scenarios` field.
