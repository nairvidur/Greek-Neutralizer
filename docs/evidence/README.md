# Evidence and status

Where each phase of the product brief stands, with what backs it.
A gate met only by proxy is reported as open.

| Phase | Gate | Status | Evidence |
|---|---|---|---|
| 1. Foundations | Clean wheel install, and a regression test per baseline defect | **Met** | `phase1/` |
| 2. Listed options analytics | Reconciled with QuantLib and a real chain | **Met**, with limitations | `phase2/` |
| 3. Optimizer and risk | Independent verification, and stress reviewed | **Met**, with limitations | `phase3/` |
| 4. Evidence | Out-of-sample report on chains, net of costs | **Open:** proxy only, needs licensed data | `phase4/` |
| 5. Product | Browser workflow, load test, restore drill, threat model | **Met on one machine**, not on a real deployment | `phase5/` |
| 6. Pilot | Shadow mode, independent validation, pen test | **Not started:** needs people and partners | |
| 7. Expansion | Futures options, FX, rates, commodities, crypto | **Not started** | |

Phase 5 went ahead with phase 4 open, because that gate depends on data that has to be purchased, and building a workflow risks no money.
That is a departure from running the phases strictly in order.

## What needs a human

| Item | Needed for |
|---|---|
| Historical option chain data license | Closing phase 4 |
| Live data agreement, including OPRA redistribution terms | Any real use |
| Design partner | Phase 6 |
| Independent model validator, and penetration tester | Phase 6 |
| Legal review of disclaimers and of adviser or broker-dealer status | Before any client use |
| ISDA SIMM license, and CCP margin files | Official-style margin |
| License model for this repository | Commercial direction |
| Who may enable live order routing | Never enabled here |

## Reviews

Two independent automated reviews made 47 findings, with some overlap, all with reproductions, and each has a test.
No human has reviewed this code.
