"""Verify the Project 2 Kaggle archive and write per-trial and aggregate tables."""
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path


def trace_id(seed):
    rng = random.Random(seed)
    trace = [
        {
            "prompt": "Explain why a database transaction matters. " * rng.choice([1, 8, 48]),
            "max_new_tokens": rng.choice([16, 32, 64]),
            "deadline_seconds": 10.0,
        }
        for _ in range(60)
    ]
    return hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def verify(path):
    trials = []
    groups = []
    with zipfile.ZipFile(path) as evidence:
        assert evidence.testzip() is None
        assert evidence.read("run-status.txt").strip() == b"completed"
        for rate in (2, 10, 30):
            for policy in ("fixed", "tokens"):
                combined = Counter()
                latencies = []
                throughputs = []
                peaks = []
                for rep in range(3):
                    data = json.loads(evidence.read(f"rate-{rate}/{rep}-{policy}.json"))
                    state = data["final_state"]
                    assert data["trace_sha256"] == trace_id(7 + rep)
                    assert data["backend"] == "VLLMBackend" and state["policy"] == policy
                    assert data["offered_requests_per_second"] == rate and data["seed"] == 7 + rep
                    assert state["healthy"] and all(state[k] == 0 for k in ("active", "queued", "reserved_tokens"))
                    if policy == "tokens":
                        assert state["peak_reserved_tokens"] <= state["budget_tokens"] == 2048
                    rows = data["requests"]
                    assert sorted(r["index"] for r in rows) == list(range(60))
                    counts = Counter(str(r["status"]) for r in rows)
                    assert counts == data["summary"]["outcomes"]
                    combined.update(counts)
                    ok = [r for r in rows if r["status"] == 200]
                    assert all(r["response"]["output_tokens"] > 0 for r in ok)
                    latencies.extend(r["latency_seconds"] for r in ok)
                    throughput = sum(r["response"]["output_tokens"] for r in ok) / data["summary"]["elapsed_seconds"]
                    assert math.isclose(throughput, data["summary"]["successful_output_tokens_per_second"])
                    throughputs.append(throughput)
                    samples = [s["board_memory_mib_utilization_percent_power_w"].split(",") for s in data["gpu_samples"] if s.get("exit_code") == 0]
                    assert samples, "missing device measurements"
                    peak = max(float(s[0]) for s in samples)
                    peaks.append(peak)
                    trials.append(dict(rate=rate, policy=policy, rep=rep, trace=data["trace_sha256"], outcomes=json.dumps(counts), success_p95_s=data["summary"]["success_p95_seconds"], tokens_per_s=throughput, peak_reserved_tokens=state["peak_reserved_tokens"], sample_count=len(samples), peak_memory_mib=peak, mean_util_pct=statistics.mean(float(s[1]) for s in samples)))
                sorted_times = sorted(latencies)
                groups.append(dict(rate=rate, policy=policy, outcomes=json.dumps(combined), pooled_success_p95_s=sorted_times[math.ceil(.95*len(sorted_times))-1] if sorted_times else None, median_trial_tokens_per_s=statistics.median(throughputs), peak_sampled_memory_mib=max(peaks)))
    write_csv(path.parent / "trials.csv", trials)
    write_csv(path.parent / "comparison.csv", groups)
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
