# Inference Gate on one Kaggle T4

On 19 November 2026, fixed concurrency and token-budget admission served the same seeded traces on one NVIDIA Tesla T4. At 10 offered requests/s, each policy completed 179 of 180 requests. Limiting outstanding reservations to 2,048 tokens reduced the peak from 2,688 while preserving that completion count. Both policies reached 8,071 MiB of sampled board memory. The run demonstrates control over gateway reservations. Physical memory savings remain unproven.

| Offered requests/s | Policy | Completed | HTTP 429 | Successful p95 (s) | Output tokens/s | Peak reserved tokens |
|---:|---|---:|---:|---:|---:|---:|
| 2 | Fixed | 180 | 0 | 1.33 | 71.7 | 960 |
| 2 | Tokens | 180 | 0 | 1.30 | 71.8 | 960 |
| 10 | Fixed | 179 | 1 | 1.94 | 293.3 | 2,688 |
| 10 | Tokens | 179 | 1 | 1.91 | 290.6 | 2,048 |
| 30 | Fixed | 93 | 87 | 2.10 | 301.7 | 2,688 |
| 30 | Tokens | 92 | 88 | 2.25 | 294.6 | 2,000 |

At 30 requests/s, nearly half of requests received HTTP 429 under either policy. The successful-request p95 excludes those rejections. Output tokens/s is the median of three trial throughputs, and peak reservations are the maximum across those trials. The small throughput differences and three repetitions give little basis for a general speed claim.

## Experiment and verification

- SmolLM2-135M-Instruct at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, vLLM 0.11.2, PyTorch 2.9.0+cu128, FP16, one Tesla T4. The engine used eager execution, a 2,048-token context limit and a 50% memory setting.
- Both policies used concurrency 8, queue capacity 8 and a 10-second request deadline. At each offered rate, three seeded traces of 60 requests were replayed under each policy. Trial order alternated and warmup requests were excluded.
- The verifier recomputed the workload hashes and checked 60 unique indices per trial, HTTP totals, output-token throughput, GPU sample presence, reservation limits, and healthy final state with zero queued or active requests and zero outstanding reservations.
- The 1,080 measured requests yielded 903 successes and 177 HTTP 429 responses for a full queue. Raw JSON retains response details, scheduling lag, latency, generated tokens on success and board telemetry. GPU samples include engine preallocation. The workload used one small model, short bursts, repeated prompt text and non-streaming generation.
- The notebook installs pip inside an isolated environment and makes Kaggle's loaded NVIDIA driver available to the FlashInfer linker. A compile-and-run probe verifies the driver before dependency installation.

[Reproduce the Kaggle run](../../docs/kaggle.md) · [Raw evidence](./inference-gate-gpu-evidence.zip) · [Per-trial CSV](./trials.csv) · [Aggregate CSV](./comparison.csv) · [Verifier](../verify.py)

Archive SHA-256 is `366eac8984c3430002b5faf474acbd0b8d3ce4de7d7779a6ee4449992222f09b`.
