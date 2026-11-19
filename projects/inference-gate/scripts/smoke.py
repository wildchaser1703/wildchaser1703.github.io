"""Exercise a running gateway and optionally require real model output."""

import argparse
import json

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8010")
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=60, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        if args.real:
            assert health.json()["backend"] != "SyntheticBackend"
        response = client.post(
            "/generate",
            json={
                "prompt": "Explain database transactions in one sentence.",
                "max_new_tokens": 48,
                "deadline_seconds": 60,
            },
        )
        response.raise_for_status()
        result = response.json()
        assert result["text"].strip() and 0 < result["output_tokens"] <= 48
        state = client.get("/state").json()
        assert state["active"] == state["queued"] == state["reserved_tokens"] == 0
        assert "gate_requests_total" in client.get("/metrics").text
        print(
            json.dumps(
                {"health": health.json(), "result": result, "state": state}, indent=2
            )
        )


if __name__ == "__main__":
    main()
