# Runbook

## Start

```sh
export GN_TOKENS="token-a=alice,token-b=bob"
uv run --extra service uvicorn --factory greeks_neutralizer.service.app:factory --port 8000
```

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
| HTTP 500 `verification_failed` | A solution failed independent repricing | **Do not trade.** Save the request and report it as a bug: this should never happen |
| Proposal status `unstable` | A small input change moves the trade and costs benefit | Review the notes, and widen the candidates or tighten the universe |
| Proposal status `within_gap` | Integer lots cost more than the configured gap | Read the `INTEGRALITY_GAP` issue. Usually fine at the stated size, and a weak bound with `max_legs` |
| 422 `infeasible` | No hedge meets the bands | Read `binding`, then widen those bands or add candidates |
| 422 `market_data` | Data failed closed | Fix the feed. Do not bypass the gate |
| 409 with `re-propose` | The proposal or its market is too old, or the market is dated in the future | Propose again on current data, after checking the feed's clock if it is in the future. `GN_MAX_MARKET_AGE_SECONDS` sets the market limit |
| `EXPIRED` in issues | A book contains an expired option | Settle or remove it |
| `EXTRAPOLATED_VOL` in issues | A strike is beyond the quotes | Treat its value and Greeks as unverified |
| Many `excluded` candidates | Quotes are stale, crossed, or empty | Check the feed before trusting the result |

## Restart and recovery

Open proposals are in memory and are lost on restart, so re-propose them.
The audit log persists and its chain continues across restarts.
Back up `GN_AUDIT_LOG` and its sidecar `GN_AUDIT_LOG.head` together, and while the service is stopped, since a log without its matching sidecar fails verification.
To test a restore, copy both files to a new path and run `verify_audit_log` on the log.

## Rotating tokens

Change `GN_TOKENS` and restart.
There is no live revocation.

## Replaying a result

A proposal's manifest holds the code version, market fingerprint, and problem fingerprint.
Run the same request on the same code version and compare the quantities and fingerprints.
The solver has a wall-clock time limit, so a run that hits it on a slower machine can return a different incumbent.
Compare objectives, not only lots, when a proposal was at its time limit.
