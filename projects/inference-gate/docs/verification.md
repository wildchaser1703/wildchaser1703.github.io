# Verification record

Verified locally on 2026-11-17 with Python 3.12, CPU PyTorch 2.5.1, and
Transformers 4.57.6. Model revision is pinned in `gate/backends.py`.

## Completed

- Fifteen tests cover budget invariants under contention, FIFO fairness, queue
  overload and expiry, cancellation before and after admission, shutdown,
  unknown backend outcomes, HTTP adapter contracts, context limits, and metrics.
- Real SmolLM2 inference returned non-empty text via HTTP. A browser submitted
  a prompt through the local Swagger UI and received HTTP 200 with model output.
- Container build and non-root synthetic HTTP smoke passed. Cloud TLS trust was
  supplied through a BuildKit secret; verification was never disabled.
- Local static checks passed. A GitHub Actions workflow includes the same tests,
  a synthetic policy experiment, container startup, and HTTP smoke checks.
- A wheel was built and installed in a fresh virtual environment outside the
  source checkout. Its local documentation assets were present and its experiment
  command completed both policy trials (8/8 synthetic requests each). This checks
  installation and package-data inclusion independently of the editable workspace.

## Recorded experiments

The checked-in `results/cpu-final` reports contain the corrected model chat
template path; both policy trials use workload hash
`aec702900249d232658773a169440e17987edbabdbc95573d7dd05167d367b4f`.

| CPU trial (one repetition, 12 requests, concurrency 1) | Fixed | Tokens |
|---|---:|---:|
| Successful requests | 12 | 12 |
| Successful requests/second | 0.577 | 0.625 |
| Successful p95 latency, seconds | 15.288 | 13.692 |
| Peak reserved tokens | 400 | 400 |

These are functional observations, not a statistically supported speedup. The
single serialized CPU worker never approached the 512-token budget. Differences
can arise from timing and load on this shared cloud machine. No GPU claim follows.

Three synthetic repetitions at 40 offered requests/second showed token-policy
peaks of 512 versus fixed-policy peaks of 1312, 880, and 1168. Fixed concurrency
completed 60/60 requests each time; the token policy completed 48, 52, and 53,
returning 429 for the remainder. The fixture has no GPU memory pressure, so extra
admission constraints reduce throughput. All raw outcomes are retained in
`results/synthetic`. Synthetic and CPU results must not be pooled.

## Kaggle GPU verification

The single-T4 run on 19 November 2026 completed all 18 trials with vLLM 0.11.2.
Every paired trace matched. All trials ended healthy with zero active requests,
queued requests and outstanding reservations. [Results and raw records](../evidence/kaggle-t4/report.md).

## Pending external verification

- Mac M4/MPS hardware execution. CPU was exercised here; MPS selection is implemented
  but requires the user's Mac to verify.


The earlier `results/cpu` directory is a development run before the chat-template
correction; it is intentionally excluded from the release and its measurements
must not be used.
