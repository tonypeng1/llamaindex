"""
Download and cache the PaulGrahamEssayDataset for benchmarking.

This uses LlamaIndex's llama-datasets hub. The dataset contains 44 GPT-4-generated
Q&A pairs with gold reference_contexts and reference_answers.

Usage:
    python -m benchmark.download_dataset          # Download / refresh
    python -m benchmark.download_dataset --check   # Just check if cached
"""

import argparse
import json
import os
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "datasets" / "paul_graham"
CACHE_FILE = CACHE_DIR / "paul_graham_qa.json"


def download_and_cache(force: bool = False) -> list[dict]:
    """
    Download PaulGrahamEssayDataset and save a simplified JSON cache.

    Each entry:
        {
            "query":              str,
            "reference_contexts": list[str],   # gold context passages
            "reference_answer":   str,          # gold answer
        }

    Returns the list of examples.
    """
    if CACHE_FILE.exists() and not force:
        print(f"✅ Dataset already cached at {CACHE_FILE}")
        return load_cached()

    print("⏳ Downloading PaulGrahamEssayDataset from llama-datasets hub …")
    try:
        from llama_index.core.llama_dataset import download_llama_dataset

        rag_dataset, _documents = download_llama_dataset(
            "PaulGrahamEssayDataset", "./data/paul_graham/llama_dataset_cache"
        )
    except Exception:
        # Fallback: if the llama-datasets hub is unavailable, try the /tmp cache
        # that was downloaded during earlier analysis.
        tmp_path = Path("/tmp/pg_rag_dataset.json")
        if tmp_path.exists():
            print(f"⚠️  Hub unavailable — falling back to {tmp_path}")
            with open(tmp_path) as f:
                raw = json.load(f)
            examples = [
                {
                    "query": ex["query"],
                    "reference_contexts": ex["reference_contexts"],
                    "reference_answer": ex["reference_answer"],
                }
                for ex in raw["examples"]
            ]
            _save(examples)
            return examples
        raise

    # Convert from LabelledRagDataset to plain dicts
    examples = []
    for ex in rag_dataset.examples:
        examples.append(
            {
                "query": ex.query,
                "reference_contexts": ex.reference_contexts,
                "reference_answer": ex.reference_answer,
            }
        )

    _save(examples)
    return examples


def _save(examples: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"✅ Saved {len(examples)} examples → {CACHE_FILE}")

    # Also write a Markdown companion for human readability
    md_path = CACHE_FILE.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(_build_dataset_markdown(examples))
    print(f"📝 Markdown companion → {md_path}")


def _build_dataset_markdown(examples: list[dict]) -> str:
    """Render the QA dataset as a readable Markdown document."""
    lines: list[str] = []
    _a = lines.append

    _a("# Paul Graham Essay — Gold QA Dataset")
    _a("")
    _a(f"**Total examples:** {len(examples)}")
    _a("")

    for idx, ex in enumerate(examples, 1):
        _a(f"---")
        _a(f"")
        _a(f"## Example {idx}")
        _a(f"")
        _a(f"### Query")
        _a(f"")
        _a(ex["query"])
        _a(f"")
        _a(f"### Reference Answer")
        _a(f"")
        _a(ex["reference_answer"])
        _a(f"")
        _a(f"### Reference Contexts")
        _a(f"")
        for ci, ctx in enumerate(ex["reference_contexts"], 1):
            if len(ex["reference_contexts"]) > 1:
                _a(f"**Passage {ci}:**")
                _a("")
            _a(ctx)
            _a("")

    _a("---")
    _a("*Generated from `benchmark.download_dataset`*")
    _a("")
    return "\n".join(lines)


def load_cached() -> list[dict]:
    """Load the cached dataset. Raises FileNotFoundError if not present."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"No cached dataset at {CACHE_FILE}. Run `python -m benchmark.download_dataset` first."
        )
    with open(CACHE_FILE) as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Paul Graham benchmark dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--check", action="store_true", help="Just check if cached")
    args = parser.parse_args()

    if args.check:
        if CACHE_FILE.exists():
            data = load_cached()
            print(f"✅ {len(data)} examples cached at {CACHE_FILE}")
        else:
            print(f"❌ No cache found at {CACHE_FILE}")
    else:
        examples = download_and_cache(force=args.force)
        print(f"\nDataset summary:")
        print(f"  Total examples:  {len(examples)}")
        print(f"  Sample query:    {examples[0]['query'][:80]}…")
        print(f"  Ref answer len:  {len(examples[0]['reference_answer'])} chars")
        print(f"  Ref context len: {len(examples[0]['reference_contexts'][0])} chars")
