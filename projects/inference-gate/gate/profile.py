"""Calculate KV cache cost from the pinned model's grouped-query geometry."""

import argparse
import json

from gate.admission import KVProfile
from gate.backends import MODEL, REVISION


def main():
    from transformers import AutoConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-mib", type=float, default=128)
    args = parser.parse_args()
    if args.kv_mib <= 0:
        parser.error("KV memory allocation must be positive")
    config = AutoConfig.from_pretrained(
        MODEL, revision=REVISION, trust_remote_code=False
    )
    profile = KVProfile(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.hidden_size // config.num_attention_heads,
    )
    budget = int(args.kv_mib * 1024**2 / profile.bytes_per_token) // 16 * 16
    print(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "kv_dtype": "float16",
                "kv_mib": args.kv_mib,
                "bytes_per_token": profile.bytes_per_token,
                "TOKEN_BUDGET": budget,
                "caveat": "KV allocation only; weights, activations and runtime overhead need separate headroom.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
