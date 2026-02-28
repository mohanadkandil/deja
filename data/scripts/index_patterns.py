"""
Index patterns from JSONL files into Qdrant knowledge graph.

Run: python -m data.scripts.index_patterns --input data/seed_patterns.jsonl
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.knowledge_graph import get_knowledge_graph
from backend.models import Pattern


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="JSONL file with patterns")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for embedding")
    parser.add_argument("--clear", action="store_true", help="Clear existing patterns first")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    kg = get_knowledge_graph()

    if args.clear:
        print("Clearing existing patterns...")
        kg.delete_all()

    # Count lines
    with open(input_path) as f:
        total = sum(1 for _ in f)

    print(f"Indexing {total} patterns from {input_path}...")

    indexed = 0
    errors = 0

    with open(input_path) as f:
        for line in tqdm(f, total=total):
            try:
                data = json.loads(line.strip())
                pattern = Pattern(**data)
                kg.add_pattern_sync(pattern)
                indexed += 1
            except Exception as e:
                print(f"Error indexing pattern: {e}")
                errors += 1

    print(f"Done! Indexed: {indexed}, Errors: {errors}")
    print(f"Knowledge graph stats: {kg.get_stats()}")


if __name__ == "__main__":
    main()
