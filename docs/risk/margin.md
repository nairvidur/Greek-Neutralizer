# Margin estimates

These are estimates for comparing alternatives.
They are not official calculations, and they must not be used to meet a margin call.

## Implemented

| Method | Function | What it leaves out |
|---|---|---|
| Reg T style short option formula | `reg_t_short_option_estimate` | Recognized spread offsets, so a spreaded book is overstated |
| Portfolio margin style price scan | `portfolio_margin_style_estimate` | Vol shocks and class offsets, so it will not match a broker's number |

The Reg T estimate ignores offsets entirely, so it does not fall when a short option is hedged.
On the SPX evidence hedge it reads the same before and after.
Use the price scan to compare a hedged book with an unhedged one.

The scan uses the FINRA Rule 4210(g) ranges as commonly described: +/-15% for equities, and +6% to -8% for broad-based index products.
Check the current rule text before relying on those ranges.

## Not implemented, deliberately

| Methodology | Why |
|---|---|
| CME SPAN and SPAN 2 | They need CME's parameter files |
| OCC STANS | The CCP's own model and inputs are needed |
| OCC TIMS official theoretical values | They need published OCC files |
| ISDA SIMM | It requires an ISDA license |

Approximating these from public descriptions would present invented numbers as margin.
Adding one requires its licensed inputs, plus reconciliation against official outputs.
