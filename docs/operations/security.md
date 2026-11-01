# Security and deployment

This describes the controls that exist today, and states what does not.
No certification is claimed: not SOC 2, ISO 27001, or anything else.

## Deployment model

Run in your own VPC or on-prem, so positions do not leave your environment.
The service makes no outbound calls.

```sh
docker build -t greeks-neutralizer .
docker run -p 8000:8000 \
  -e GN_TOKENS="$TOKEN_A=alice,$TOKEN_B=bob" \
  -v gn-data:/home/gn/data \
  greeks-neutralizer
```

Terminate TLS at a reverse proxy, since the service speaks plain HTTP.
Keep `/healthz` and `/metrics` off the public path, because they are unauthenticated.

## Controls that exist

| Control | Implementation |
|---|---|
| Authentication | Bearer tokens with one identity per token, compared in constant time |
| Refuses to run open | No tokens configured means no start |
| Four-eyes approval | The proposer cannot approve their own proposal |
| Approval expiry | A proposal older than 5 minutes cannot be approved or exported |
| Unstable block | Proposals flagged unstable cannot be approved unless policy allows it |
| Audit trail | Hash-chained JSON Lines, written and synced before each state change, with a sidecar for the head. A broken chain makes `/healthz` return 503, and blocks approvals and exports |
| Market age | A proposal on a market older than `GN_MAX_MARKET_AGE_SECONDS` cannot be submitted, approved, or exported |
| Stability required | A proposal with legs must have run its stability check to be submitted |
| Input validation | Unknown fields rejected, and numbers checked at construction |
| Request size | Limited by `GN_MAX_BODY` |
| No order routing | The package has no way to send an order |
| Dependency audit | `pip-audit` in CI |
| Container | Built from the wheel, runs non-root, with no dev dependencies |

## Not implemented

These are required before a regulated production deployment:

- **SSO, OIDC, and RBAC.** Tokens are static, with no roles beyond four-eyes.
- **Tenant isolation.** Run one deployment per team.
- **Durable proposal store.** Open proposals are in process memory and are lost on restart, though the audit log survives.
- **Write-once audit storage.** The hash chain and sidecar detect tampering but cannot prevent it, and someone able to rewrite both files would pass unless the head hash is anchored elsewhere.
- **Encryption at rest.** Rely on the volume's encryption.
- **Secrets management.** Tokens arrive through the environment.
- **Rate limiting, Helm charts, and Terraform.**
- **SBOM, signed releases, and SLSA provenance.**
- **Independent penetration test.**

## Threat model

| Threat | Current mitigation | Residual risk |
|---|---|---|
| Unauthorized hedge submission | Bearer auth | A leaked token impersonates its owner until rotated, which needs a restart |
| Self-approved trade | Four-eyes on identity | One person with two tokens defeats it |
| Trading on stale analysis | Approval expiry and a market fingerprint in the manifest | Five minutes is long in a fast market, so set it for your desk |
| Silent wrong hedge | Independent repricing, and raising on mismatch | Shared model assumptions are not caught by repricing |
| Bad market data | Quality gates fail closed | A plausible but wrong quote passes |
| Audit tampering | Hash chain checked on health | See above |
| Malicious JSON | Strict schema and a size limit | A large in-limit problem can use CPU for the solver's time limit |
| Browser injection | All rendered values are escaped | The token lives in page memory, and there is no CSP header yet |

## Regulatory notes

Citations were current when written and need checking against primary sources, with counsel where it matters.

- **Model risk:** SR 26-2 replaced SR 11-7 in April 2026, with OCC Bulletin 2026-13 as its companion. See `docs/governance/model-pack.md`.
- **Market access:** if a client connects tickets to a market, SEC Rule 15c3-5 duties are the client's. The exported tickets do not carry those controls.
- **Investment advice:** this is decision support, and whether a deployment raises adviser or broker-dealer questions is a legal question to resolve first.
