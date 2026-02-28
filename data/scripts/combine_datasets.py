"""
Combine multiple pattern JSONL files into a single seed file.

Run: python -m data.scripts.combine_datasets
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=["data/nvidia_patterns.jsonl", "data/swebench_patterns.jsonl"],
        help="Input JSONL files"
    )
    parser.add_argument("--output", type=str, default="data/seed_patterns.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    seen_ids = set()

    with open(output_path, "w") as out:
        for input_file in args.inputs:
            input_path = Path(input_file)
            if not input_path.exists():
                print(f"Warning: {input_path} not found, skipping")
                continue

            count = 0
            with open(input_path) as f:
                for line in f:
                    data = json.loads(line.strip())
                    pattern_id = data.get("pattern_id", "")

                    # Dedupe by pattern_id
                    if pattern_id in seen_ids:
                        continue
                    seen_ids.add(pattern_id)

                    out.write(json.dumps(data) + "\n")
                    count += 1

            print(f"Added {count} patterns from {input_path}")
            total += count

    print(f"Total: {total} patterns saved to {output_path}")


if __name__ == "__main__":
    main()
