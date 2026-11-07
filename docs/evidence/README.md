# Evidence and status

Where each phase of the product brief stands, with what backs it.
A gate met only by proxy is reported as open.

| Phase | Gate | Status | Evidence |
|---|---|---|---|
| 1. Foundations | Clean wheel install, and a regression test per baseline defect | **Met** | `phase1/` |
| 2. Listed options analytics | Reconciled with QuantLib and a real chain | **Met**, with limitations | `phase2/` |
| 3. Optimizer and risk | Independent verification, and stress reviewed | **Met**, with limitations | `phase3/` |
| 4. Evidence | Out-of-sample report on chains, net of costs | **Met on free chains.** No-go for the optimizer as first configured; its position-aware fix is a go on held-out DIA, not confirmed on XLF, at a measured cost in carry; daily delta stays the default | `phase4/` |
| 5. Product | Browser workflow, load test, restore drill, threat model | **Met on one machine**, not on a real deployment | `phase5/` |
| 6. Pilot | Shadow mode, independent validation, pen test | **Not started:** needs people and partners | |
| 7. Expansion | Futures options, FX, rates, commodities, crypto | **Not started** | |

Phase 5 went ahead with phase 4 open, because that gate then seemed to need purchased data, and building a workflow risks no money.
That is a departure from running the phases strictly in order.
Phase 4 has since been run on free historical chains.
Its no-go led to a position-aware fix, which passed a pre-registered held-out test on DIA and was not confirmed on XLF, so it is eligible for the phase 6 paper pilot and is not a default.

## What needs a human

| Item | Needed for |
|---|---|
| Licensed option chain with every listed strike | Marking held positions at observed prices, and rerunning phase 4 after the optimizer is fixed |
| Live data agreement, including OPRA redistribution terms | Any real use |
| Design partner | Phase 6 |
| Independent model validator, and penetration tester | Phase 6 |
| Legal review of disclaimers and of adviser or broker-dealer status | Before any client use |
| ISDA SIMM license, and CCP margin files | Official-style margin |
| License model for this repository | Commercial direction |
| Who may enable live order routing | Never enabled here |

## Reviews

The first two independent automated reviews made 47 findings, with some overlap, all with reproductions, and each has a test.
Later automated review rounds on this branch made more findings, which are not counted in that figure, with tests in `tests/test_third_review.py`, `tests/test_fourth_review.py`, and `tests/test_fifth_review.py`.
No human has reviewed this code.
