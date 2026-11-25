# Inference Gate

How much work should an LLM server accept before its queue becomes the problem?

Inference Gate puts a bounded admission queue in front of a single inference engine.
It counts prompt tokens, reserves space for the maximum output, and compares that
policy against a fixed concurrency limit using the same open-loop request trace.
The gateway exposes its decisions, deadlines, and occupancy for inspection.

## What happened on a T4?

At 10 offered requests per second, both policies completed 179 of 180 requests.
The token budget cut peak reservations from 2,688 to 2,048 tokens. Peak sampled
GPU memory was 8,071 MiB for both. At 30 requests per second, both queues filled
and nearly half the requests received HTTP 429.

The experiment covers 1,080 requests across 18 trials on one Kaggle Tesla T4
running vLLM and SmolLM2-135M-Instruct. The report includes every rejection,
successful-request latency and throughput. [Read the results and download the evidence](evidence/kaggle-t4/report.md).

## Run on your M4 Mac (24 GB)

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[local,dev]'
BACKEND=transformers MAX_CONCURRENCY=1 uvicorn gate.api:app --host 127.0.0.1 --port 8010
```

The first start downloads the public, Apache-2.0 SmolLM2-135M-Instruct model at a
pinned revision. No API key or paid service is used. CUDA, then MPS, then CPU is
selected automatically. This small model demonstrates real inference; its answer
quality is not the subject of the experiment. HF generation is serialized; GPU
concurrency measurements use vLLM. Keep a single gateway process (`--workers 1`).

Open `http://127.0.0.1:8010/docs` for the interactive API, or:

```sh
curl http://127.0.0.1:8010/generate -H 'content-type: application/json' \
  -d '{"prompt":"Explain database transactions in one sentence.","max_new_tokens":32,"deadline_seconds":30}'
curl http://127.0.0.1:8010/state
curl http://127.0.0.1:8010/metrics
```

## Test without downloading model weights

```sh
pip install -e '.[dev]'
pytest -q
python -m gate.experiment --backend synthetic --count 60 --rate 40
```

Or `docker compose up -d --build --wait` starts the **synthetic** API on localhost
port 8010. `/health` and every benchmark report identify the active backend.

## Reproduce the comparison

```sh
python -m gate.experiment --backend transformers --concurrency 1 --count 12 --rate 2 --repetitions 1
python -m gate.profile --kv-mib 128
```

`gate.profile` derives the FP16 KV cost from the model's actual attention geometry.
Choose a KV allocation after accounting for model weights and runtime overhead,
then use its `TOKEN_BUDGET` with the vLLM adapter. This is a static admission estimate,
not a promise of preventing every CUDA out-of-memory error.

Reports include raw request outcomes, workload hashes, offered rate, completed
throughput, successful-request latency percentiles, and peak token reservations.
Repetitions alternate policy order. Rejecting requests can improve latency while
reducing useful work: always compare outcome counts as well as latency.

See [the Kaggle runbook](docs/kaggle.md), [design tradeoffs](docs/design.md), and
[verification record](docs/verification.md).

## API behavior

| Situation | Response / behavior |
|---|---|
| Available capacity | Generate and return text plus token counts and timing |
| Queue full | 429 with Retry-After |
| Context or reservation too large | 413 |
| Invalid input | 422 |
| Queue or execution deadline elapsed | 504 |
| Client times out during generation | Keep reservation until backend finishes |
| Unknown remote outcome | 503; stop new admissions until operator checks engine and restarts gateway |

Local research service: no authentication, public ingress, paid dependencies,
distributed admission, streaming, or production availability claim. Queued requests
are in memory and are lost on restart; this is intentionally different from the
durable workflow service. Do not route other traffic directly to the same engine
when evaluating the gateway's budget.
