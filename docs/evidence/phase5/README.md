# Phase 5 evidence: product

Gate: "end-to-end trader workflow in a browser, load test, restore drill, and threat model complete."

**Status: partial.**

| Item | Status |
|---|---|
| End-to-end workflow in a real browser | **Done**, see below |
| Container built and run | **Done:** starts as uid 10001, enforces auth, and returned a verified proposal over HTTP |
| Threat model | **Done** as a first version, in `docs/operations/security.md`. Not independently reviewed |
| Audit log restart and integrity | **Tested** in `tests/`, including tampering and truncation |
| Restore drill | **Not done** as a drill on a real deployment |
| Load test | **Not done** |

## Browser run

Driven with Chrome through `chrome-devtools-axi`, against the running service:

1. Alice connects, loads the sample, and proposes a hedge: `ui_proposal_desktop.png`.
2. Alice submits it and tries to approve her own proposal, and four-eyes refuses it: `ui_four_eyes_refused.png`.
3. Bob connects and approves it: `ui_approved.png`.
4. At 390 by 844, the page has no horizontal overflow: `ui_phone.png`.

The audit chain stayed intact, and the only console error was the expected 409 from step 2.
The sample uses illustrative values, not market data.

Browser testing found issues the test suite could not: a favicon 404, a token field outside a form, a button enabled before it could work, misaligned table headers, a duplicate status badge, and "-0" in the grid.
It also exposed the stability defect, by blocking an approval that should have gone through.

## Not done

Excel add-in, OMS and PMS import, FIX drop copy, broker APIs, paper or live execution, streaming updates, SSO, Helm, and Terraform.
See `docs/limitations.md`.
