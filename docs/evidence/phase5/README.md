# Phase 5 evidence: product

Gate: "end-to-end trader workflow in a browser, load test, restore drill, and threat model complete."

**Status: met, on one machine and a local disk.**
Nothing here has run on a real deployment, and the threat model has had no independent review.

| Item | Status |
|---|---|
| End-to-end workflow in a real browser | **Done**, see below |
| Container built and run | **Done:** runs as uid 10001 with auth enforced. A proposal submitted over HTTP wrote its audit log to a named volume, and the chain stayed intact after the container was replaced |
| Threat model | **Done** as a first version, in `docs/operations/security.md`. Not independently reviewed |
| Audit log restart and integrity | **Tested** in `tests/`, including tampering and truncation |
| Restore drill | **Done** against real service processes, see below. No remote backup target or real power loss |
| Load test | **Done** against a real service process, see below. Sample request only, on loopback |

## Browser run

Driven with Chrome through `chrome-devtools-axi`, against the running service:

1. Alice connects, loads the sample, and proposes a hedge: `ui_proposal_desktop.png`.
2. Alice submits it, opens it, and tries to approve her own proposal, and four-eyes refuses it: `ui_four_eyes_refused.png`.
3. Bob connects, where Approve is disabled until he opens the proposal: `ui_review_before_approval.png`.
4. Bob opens it, which renders its status, issues, legs, residuals, and exclusions, and approves it: `ui_approved.png`.
5. Under 390 by 844 viewport emulation, the page has no horizontal overflow: `ui_phone.png`.

The audit chain stayed intact, and the only console error was the expected 409 from step 2.
The sample uses illustrative values, not market data.

Browser testing found issues the test suite could not: a favicon 404, a token field outside a form, a button enabled before it could work, misaligned table headers, a duplicate status badge, and "-0" in the grid.
It also exposed the stability defect, by blocking an approval that should have gone through.

## Load test

`scripts/load_test.py`, with results in `load_test.json`.
One uvicorn worker on a 12-core laptop, with the load generator on the same machine, and 20 seconds per case.
The request is the small sample proposal, not a real chain.

| Endpoint | 1 client | 4 clients | 16 clients | p99 at 16 clients |
|---|---:|---:|---:|---:|
| `/v1/value` | 727/s | 1,097/s | 1,171/s | 27 ms |
| `/v1/stress` | 498/s | 634/s | 654/s | 37 ms |
| `/v1/hedge`, with stability | 5.0/s | 12.9/s | 12.3/s | 1,808 ms |

Clients wait for each response here, so these rows measure capacity, and they understate latency near saturation.
Hedging saturates near 13 per second and gains nothing from more clients.

At a fixed arrival rate, timed from each request's scheduled start so that a stall counts against everything behind it, hedging held up to the highest rate offered:

| Offered | p50 | p99 | Max |
|---:|---:|---:|---:|
| 1/s | 252 ms | 277 ms | 282 ms |
| 2/s | 249 ms | 254 ms | 254 ms |
| 4/s | 206 ms | 221 ms | 248 ms |
| 8/s | 231 ms | 250 ms | 275 ms |

**Nothing was lost under load.**
500 concurrent submissions all returned 201, and each was then raced by two approvers.
Every race had exactly one 200 and one 409, and the log held exactly 500 `submitted` and 500 `approved` records, with its chain intact.
There were no 5xx responses in any case.

`GET /v1/proposals` was not load tested.
Its summaries and `limit` postdate this measurement, and so do the solver gates, deployment defaults for gates, and the risk report endpoints, none of which the requests above use.

Limits this exposed:

- **A real chain is far slower.** The 2,254-candidate SPX hedge takes about 5 seconds, so expect well under one per second per worker. It was not load tested, since the vendor chain is not committed.
- **Audit verification grows with the log.** Every submission, approval, rejection, and export, and every `/healthz`, rereads the whole log under its lock. `/healthz` took 7 ms at 1,020 records and 40 ms at 5,020, and a submission's p50 went from 213 ms to 244 ms. Verification is what detects tampering, so it stays complete for now. Beyond these sizes the cost is an extrapolation of about 8 ms per 1,000 records, not a measurement.
- **One worker only.** Proposals and the audit lock live in one process, so a second worker would split the store and fork the chain. This limits scaling to one machine's single process until the store moves out of process.

## Restore drill

`scripts/restore_drill.py`, with results in `restore_drill.json`, following `docs/operations/runbook.md` and failing if any step behaves differently.

- **Restore.** 40 records written over HTTP, backed up with checksums while stopped, and the originals destroyed. The restored service was healthy 0.67 seconds after the copy began, with all 40 records. A new submission extended the chain to 41, and it survived another restart.
- **Damaged backups.** A missing sidecar, a sidecar from an earlier time, a truncated log, and an edited record each returned 503 `audit_chain_broken` from `/healthz` and 409 on submission.
- **Crashes.** SIGKILL during 8 concurrent submitters, 5 times. No acknowledged submission was lost in any run, 56 in total.
- **1 of the 5 crashes left the log ahead of its sidecar**, since the kill landed between the two writes. The service then refuses work until an operator follows the runbook. That is fail-closed behavior and intended, but it means a hard crash can need a person. Open proposals do not survive any restart, by design, so they have to be proposed again.

Not exercised: a remote backup target, a real power loss, or a disk that misreports fsync.

## Not done

Excel add-in, OMS and PMS import, FIX drop copy, broker APIs, paper or live execution, streaming updates, SSO, Helm, and Terraform.
See `docs/limitations.md`.
