"""
Transform SWE-bench Lite dataset into Pattern format.

Run: python -m data.scripts.transform_swebench
"""

import json
import argparse
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import httpx
import uuid
from datetime import datetime

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

TRANSFORM_PROMPT = """You are a pattern extraction engine. Given a GitHub issue and its patch, extract a standardized debugging pattern.

Return a JSON object with these exact fields:
{
    "problem_class": "Short abstract name (e.g., 'Django ORM query optimization')",
    "problem_signature": "1-2 sentence abstract description of the bug pattern",
    "domain_tags": ["tag1", "tag2", "tag3"],
    "key_insight": "The core realization that leads to the fix",
    "solution_template": "Step-by-step approach to fix this type of bug",
    "difficulty": "easy|medium|hard",
    "estimated_token_savings": 10000-30000
}

Return ONLY valid JSON, no other text."""


def call_mistral_sync(problem: str, patch: str, repo: str) -> dict:
    response = httpx.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistral-small-latest",
            "messages": [
                {"role": "system", "content": TRANSFORM_PROMPT},
                {"role": "user", "content": f"Repository: {repo}\n\nIssue:\n{problem}\n\nPatch:\n{patch}"}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        },
        timeout=60.0
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def extract_repo_tags(repo: str) -> list[str]:
    """Extract domain tags from repo name."""
    repo_lower = repo.lower()
    tags = []

    if "django" in repo_lower:
        tags.extend(["django", "web", "python", "orm"])
    elif "flask" in repo_lower:
        tags.extend(["flask", "web", "python", "api"])
    elif "scikit" in repo_lower or "sklearn" in repo_lower:
        tags.extend(["scikit-learn", "ml", "python", "data-science"])
    elif "matplotlib" in repo_lower:
        tags.extend(["matplotlib", "visualization", "python", "plotting"])
    elif "sympy" in repo_lower:
        tags.extend(["sympy", "math", "python", "symbolic"])
    else:
        tags.extend(["python", "open-source"])

    return tags


def transform_entry(entry: dict) -> dict:
    problem = entry.get("problem_statement", "")
    patch = entry.get("patch", "")
    repo = entry.get("repo", "")
    instance_id = entry.get("instance_id", "")

    try:
        extracted = call_mistral_sync(problem[:3000], patch[:2000], repo)
    except Exception as e:
        print(f"Error calling Mistral for {instance_id}: {e}")
        extracted = {
            "problem_class": f"Bug in {repo}",
            "problem_signature": problem[:200],
            "domain_tags": extract_repo_tags(repo),
            "key_insight": "See patch for fix",
            "solution_template": "Apply the patch to fix the issue",
            "difficulty": "medium",
            "estimated_token_savings": 15000
        }

    all_tags = list(set(extracted.get("domain_tags", []) + extract_repo_tags(repo)))

    pattern = {
        "pattern_id": f"SWE-{instance_id.replace('__', '-')[:20]}",
        "problem_class": extracted["problem_class"],
        "problem_signature": extracted["problem_signature"],
        "domain_tags": all_tags,
        "reasoning_trace": {
            "failed_approaches": [],
            "key_insight": extracted["key_insight"]
        },
        "solution_template": extracted["solution_template"],
        "generic_code_template": patch[:2000] if patch else None,
        "metadata": {
            "source": "swe_bench",
            "verification_status": "verified",
            "success_rate": 1.0,  # Ground truth patches
            "times_applied": 0,
            "estimated_token_savings": extracted.get("estimated_token_savings", 15000),
            "difficulty": extracted["difficulty"]
        },
        "refinements": [],
        "created_at": datetime.utcnow().isoformat(),
        "original_instance_id": instance_id,
        "original_repo": repo
    }

    return pattern


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/swebench_patterns.jsonl")
    parser.add_argument("--holdout", type=str, default="data/eval_holdout.jsonl")
    parser.add_argument("--holdout-count", type=int, default=50, help="Number of patterns to hold out for evaluation")
    parser.add_argument("--skip-transform", action="store_true", help="Skip Mistral transformation")
    args = parser.parse_args()

    print("Loading SWE-bench Lite dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    output_path = Path(args.output)
    holdout_path = Path(args.holdout)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_patterns = []

    print(f"Processing {len(dataset)} entries...")
    for entry in tqdm(dataset):
        if args.skip_transform:
            repo = entry.get("repo", "")
            pattern = {
                "pattern_id": f"SWE-{entry.get('instance_id', '')[:20]}",
                "problem_class": f"Bug in {repo}",
                "problem_signature": entry.get("problem_statement", "")[:300],
                "domain_tags": extract_repo_tags(repo),
                "reasoning_trace": {
                    "failed_approaches": [],
                    "key_insight": "Apply the ground truth patch"
                },
                "solution_template": "See patch for solution",
                "generic_code_template": entry.get("patch", "")[:2000],
                "metadata": {
                    "source": "swe_bench",
                    "verification_status": "verified",
                    "success_rate": 1.0,
                    "times_applied": 0,
                    "estimated_token_savings": 15000,
                    "difficulty": "medium"
                },
                "refinements": [],
                "created_at": datetime.utcnow().isoformat(),
                "original_instance_id": entry.get("instance_id", ""),
                "original_repo": repo
            }
        else:
            pattern = transform_entry(entry)

        all_patterns.append(pattern)

    holdout_patterns = all_patterns[:args.holdout_count]
    training_patterns = all_patterns[args.holdout_count:]

    with open(output_path, "w") as f:
        for pattern in training_patterns:
            f.write(json.dumps(pattern) + "\n")
    print(f"Saved {len(training_patterns)} patterns to {output_path}")

    with open(holdout_path, "w") as f:
        for pattern in holdout_patterns:
            f.write(json.dumps(pattern) + "\n")
    print(f"Saved {len(holdout_patterns)} holdout patterns to {holdout_path}")


if __name__ == "__main__":
    main()
