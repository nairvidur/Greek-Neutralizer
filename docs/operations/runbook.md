# Runbook

## Start

```sh
export GN_TOKENS="token-a=alice,token-b=bob"
uv run --extra service uvicorn --factory greeks_neutralizer.service.app:factory --port 8000
```

**Run exactly one worker.**
Proposals and the audit lock live in one process, so a second worker would split the store and fork the audit chain.
`docs/evidence/phase5/README.md` has the measured capacity.

Open `http://127.0.0.1:8000/` and enter a token.

## Health

`GET /healthz` returns 200 with `ok`, or 503 with `audit_chain_broken`.
`GET /metrics` returns request and error counts in Prometheus text format.
Each request logs one JSON line with its request ID, path, status, and duration.

## Responding to problems

| Symptom | Meaning | Action |
|---|---|---|
| `audit_chain_broken` | The audit log or its sidecar was edited, truncated, or deleted, or a write was interrupted | **Treat as an incident.** Submissions, approvals, rejections, and exports are already blocked. Preserve both files, and read `first_bad_line` and `reason` from `/v1/audit` |
| `reason` says the log is longer than its head sidecar | A record was committed but the sidecar was not updated: a crash or a failed write, or records were added | Compare the extra records with what the service did. The action they describe may have reported failure. Keep both files as the record, and point `GN_AUDIT_LOG` at a new path to resume |
| Proposal status `no_usable_candidates` | Every candidate was excluded | **It does not mean no hedge is needed.** Fix the feed, then re-propose |
| Proposal status `partial_coverage` | A targeted underlying has no candidate that can trade, and is unhedged | **Read the `NO_USABLE_CANDIDATES` issue.** Its residual stays as it was, so fix the feed or add candidates, then re-propose |
| 422 `market_data`, "model and quotes disagree" | Most quotes are far from the model | Check vol units, rates, and spot before anything else |
| HTTP 500 `verification_failed` | A solution failed independent repricing, or a candidate at capacity beat the solver's own lower bound | **Do not trade.** Save the request and report it as a bug: this should never happen |
| Proposal status `unstable` | A small input change moves the trade and costs benefit | Review the notes, and widen the candidates or tighten the universe |
| Proposal status `within_gap` | Integer lots cost more than the configured gap | Read the `INTEGRALITY_GAP` issue. Usually fine at the stated size, and a weak bound with `max_legs` |
| 422 `infeasible` | The solver proved, over whole lots and shares at full capacity, that no hedge meets the bands | Read `binding`, then widen those bands or add candidates |
| 422 `validation`, "looser than its limit" | The request loosens a gate past this service's limit | Leave the gate out to get the limit, or have whoever owns the deployment raise that `GN_GATE_*` limit. Do not change it to get a trade through |
| `GATE_SETTINGS` in issues | The request set a fail-closed gate differently from this service's limit | Approvers: decide whether that setting is justified before approving |
| 422 `search_limit` | The search stopped with no hedge, in the base solve or a stability re-solve. **This does not mean no hedge exists** | Reduce the universe. A request can raise `mip_node_limit` or `time_limit_seconds` only up to `GN_GATE_MIP_NODE_LIMIT` and `GN_GATE_TIME_LIMIT_SECONDS`, so past that it is the deployment owner's decision. Do not widen bands because of it |
| 422 `market_data` | Data failed closed | Fix the feed. Do not bypass the gate |
| 409 with `re-propose` | The proposal or its market is too old, or the market is dated in the future | Propose again on current data, after checking the feed's clock if it is in the future. `GN_MAX_MARKET_AGE_SECONDS` sets the market limit |
| `EXPIRED` in issues | A book contains an expired option | Settle or remove it |
| `EXTRAPOLATED_VOL` in issues | A strike is beyond the quotes | Treat its value and Greeks as unverified |
| Many `excluded` candidates | Quotes are stale, crossed, or empty | Check the feed before trusting the result |
| `PROXY_MARK` in issues | A proxy is valued at spot times its delta ratio, not its market price | Read its reported value with care. Its delta and scenario P&L are right |
| `QUOTE_UNCHECKED` in issues | A proxy leg's quote has no market reference to compare with | Approvers: check that quote against its own market before approving |
| `excluded` reason "the market implies" | A stock or future quote is more than `max_linear_quote_gap` from spot times its ratio, or from the forward | Check spot, the symbol, and the quote's units. Do not raise `GN_GATE_MAX_LINEAR_QUOTE_GAP` to get a trade through |
| `UNCONVERGED_FIT` in a chain script's issues | An expiry's fit reached its evaluation limit, as SciPy reports it | Read that expiry's fit errors before relying on its part of the surface |

## Gate limits

Each `GN_GATE_*` setting is that gate's limit, listed in `src/greeks_neutralizer/service/app.py`.
A request that omits a gate gets the limit, so tightening one does not break clients or the sample.
A limit looser than the library default therefore loosens every request that omits the gate, with no `GATE_SETTINGS` issue, so raising one is a risk decision.
A proposal's `solver_gap` shows how far the hedge may be from the LP bound, whatever status it carries.

## Risk reports

These reprice from the request and store nothing:

| Endpoint | CLI | Reports |
|---|---|---|
| `POST /v1/ladder` | `gn ladder` | Option and dated future exposures by New York expiry date, so each date shows the net position. Shares are left out, because they do not expire |
| `POST /v1/limits` | `gn limits` | Which limits in the request the book breaches, of how many checked |
| `POST /v1/liquidity` | `gn liquidity` | Half-spread cost and days to exit, from a spread and daily volume the caller supplies |
| `POST /v1/decay` | `gn decay` | One underlying's exposures as time passes, with spot and vol unchanged |

A breach is a line in a response, and nothing acts on it: see `docs/limitations.md`.
Each of these except liquidity, and `/v1/stress`, `/v1/grid`, `/v1/frontier`, `gn stress`, and `gn frontier`, carries the book's valuation `issues`, so read them as you would on a proposal: an expired option shows no risk here, and is still there to settle.
`gn decay` and `/v1/decay` also list an issue first appearing at a step, marked with that step, such as `(at +10d)`.

A limit needs a `lower`, an `upper`, or both, and a bound must be finite.
`utilization` on a breach is the distance from the band's centre over its half-width, so above 1.0 means breached: 10,000 against [-5000, 5000] is 2.0, and 1 against [100, 1000] is about 1.22.
A zero-width band, or one with a side left out, has no such ratio, so `utilization` is null and `utilization_unbounded` is true.

A scenario, or a decay step, can look forward at most 36,525 days.
Every date in a request must fall from 1900 to 2300.
A `carry_yield` above 1.0 either way is refused, since rates are decimals, and so is a spot above 1e9, a position above 1e12 units, or an objective cost above 1e15.
A scenario that would take spot past 1e9 is refused too, by name.
A hedge request is also refused when a scenario loss, band, or budget reaches 1e20, which the solver reads as infinite, or when an instrument too small for the solver to see could still make $1,000 or more at capacity under a typical shock for that Greek.
Below that it is not refused: the solver just does not see its effect on that row, and verification reprices whatever results.
A refusal says nothing about whether a hedge exists, so it is never reported as infeasible.
A stability re-solve refused this way marks the proposal unstable, and a frontier point refused this way has status `out_of_range`, while the other points stand.

## Listing proposals

`GET /v1/proposals` returns summaries, newest first, 50 by default and at most 500 with `limit`.
`total` against `count` shows when the list was cut.
`GET /v1/proposals/{id}` returns the full proposal, which is what the UI's Open fetches.

## Restart and recovery

Open proposals are in memory and are lost on restart, so re-propose them.
The audit log persists and its chain continues across restarts.
Back up `GN_AUDIT_LOG` and its sidecar `GN_AUDIT_LOG.head` together, and while the service is stopped, since a log without its matching sidecar fails verification.
To test a restore, copy both files to a new path and run `verify_audit_log` on the log.
`scripts/restore_drill.py` runs the whole procedure against real service processes.

A hard crash can land between the log and sidecar writes, which happened in 1 of 5 kills in the drill.
The service then reports `audit_chain_broken` and refuses work until someone follows the row for it above.
No acknowledged action was lost in the drill.

Every audited action and every `/healthz` rereads the whole log, at about 8 ms per 1,000 records measured up to 5,000.
When that matters, stop the service, keep both files as the record, and point `GN_AUDIT_LOG` at a new path.
The new chain starts from nothing, so keep the old one with its checksums, since nothing links the two.

## Rotating tokens

Change `GN_TOKENS` and restart.
There is no live revocation.

## Replaying a result

A proposal's manifest holds the code version, SciPy and NumPy versions, stability flag, market fingerprint, and problem fingerprint.
Run the same request on the same code and library versions, and compare the quantities and fingerprints.
The search stops at a node count (`mip_node_limit`), so machine load does not change the result.
What was measured is two byte-identical runs on one machine, one of them under load.
Other library versions can change the result, and other architectures are unverified.
The wall-clock limit is a backstop only: a run where any solve reaches it carries a `SOLVER_TIME_LIMIT` issue, and it will not replay exactly.
