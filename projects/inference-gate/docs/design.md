# Design and tradeoffs

## Boundary

Client → tokenization → bounded FIFO admission → inference engine → response.
The gateway controls admission, not GPU kernels, continuous batching, or paged
attention. vLLM owns those. No inference-engine reinvention is claimed here.

## Reservation model

For standard attention with GQA, KV bytes per token =
`2 × layers × KV heads × head dimension × bytes per element`.
The first factor is key plus value. Use KV heads, not query heads. The gateway
reserves prompt plus maximum output, rounded up to a 16-token block. vLLM may
allocate lazily; reserving the worst case gives less utilization but predictable
admission. Prefix caching and early EOS can make that estimate conservative.
Weights, activations, temporary buffers, CUDA graphs, fragmentation, and other
users of the GPU are outside this estimate. The fixed baseline uses the same
queue/concurrency/deadline settings but ignores aggregate token occupancy.

## Queue policy

FIFO deliberately prevents a stream of short requests from starving a long one.
Its cost is head-of-line blocking: a large first request can hold back a smaller
request that would fit. A size-aware or aging policy is a future controlled
experiment, not an unmeasured improvement. Queue size and deadlines bound memory
and waiting. Concurrency is independently capped even when token capacity remains.

## Cancellation and unknown outcomes

Execution runs in a tracked task protected from client cancellation. A client
deadline releases the HTTP caller, not GPU capacity. The task releases its
reservation only when generation returns. This wastes work when a client leaves,
but avoids assuming an HTTP disconnect cancels remote computation. A transport
failure can leave upstream work alive; the admission circuit closes on backend
errors. Recovery requires checking the engine and restarting the gateway. No
automatic retry risks doubling model work. Graceful shutdown stops new admission,
wakes waiters, and drains tracked work.

## One coordinator

The lock protects state in one asyncio process. Several uvicorn workers would
oversubscribe the same budget. Scaling requires a coordinator per inference
replica and routing affinity, or atomic reservations in a shared store with
leases. That is outside the demonstrated single-replica experiment.

## Measurement

An open-loop client schedules requests by time rather than waiting for completions.
This exposes overload instead of automatically reducing offered load when the
server slows. Reports retain scheduling lag, every HTTP status, output tokens,
successful-request latency and throughput. Timeout/rejection counts must accompany
latency percentiles to avoid survivorship bias. Workloads are seeded, hashed, and
identical between policies. Warmup is excluded; trial order alternates.

The default trace has mixed prompt lengths and output caps. A lower reserved-token
peak does not establish lower measured VRAM or higher throughput. The [T4 report](../evidence/kaggle-t4/report.md) compares both policies on identical traces. Non-streaming generation cannot measure time-to-first-token.
