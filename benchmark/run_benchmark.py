"""
Benchmark runner: Paul Graham Essay RAG evaluation.

Runs every query from the gold dataset through:
  1. The full pipeline (MinerU + GLiNER + LangExtract + BM25/vector fusion +
     ColBERT reranking + SubQuestionQueryEngine)
  2. A vanilla LlamaIndex baseline (same docs, default chunking, simple
     VectorStoreIndex query engine, no reranking or entity filtering)

For each, computes:
  • Context Recall  — sentence-level overlap with gold reference contexts
  • Faithfulness    — LLM-as-judge (1–5)
  • Correctness     — LLM-as-judge (1–5)

Outputs a summary table to stdout and a timestamped JSON report to
benchmark/results/.

Usage:
    python -m benchmark.run_benchmark                       # full run (44 queries × 2 pipelines)
    python -m benchmark.run_benchmark --limit 5              # first 5 queries
    python -m benchmark.run_benchmark --start 4 --limit 3    # queries 4-6
    python -m benchmark.run_benchmark --skip-baseline        # pipeline only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from benchmark.download_dataset import download_and_cache
from benchmark.evaluators.retrieval_eval import compute_context_recall
from benchmark.evaluators.llm_judge import evaluate_faithfulness, evaluate_correctness

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Vanilla baseline builder
# ---------------------------------------------------------------------------

def _build_vanilla_baseline() -> Any:
    """
    Build an in-memory vanilla LlamaIndex pipeline.

    Same LLM and embedding model as the full pipeline, but:
      - Default 1024-token chunks (LlamaIndex default)
      - Simple VectorStoreIndex in memory (no Milvus/Mongo)
      - Plain RetrieverQueryEngine (no SubQuestionQueryEngine)
      - No BM25 fusion, no ColBERT reranking, no entity filtering
    """
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.anthropic import Anthropic

    from config import EMBEDDING_CONFIG

    # Same LLM/embeddings as the full pipeline
    llm = Anthropic(
        model="claude-sonnet-4-5",
        temperature=0.0,
        max_tokens=2500,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    embed_model = OpenAIEmbedding(
        model_name=EMBEDDING_CONFIG["model_name"],
        embed_batch_size=10,
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Load documents from the existing MinerU output (no re-parsing needed)
    from main import load_document_mineru

    content_list_path = (
        PROJECT_ROOT / "data" / "paul_graham" / "mineru_output"
        / "paul_graham_essay" / "vlm" / "paul_graham_essay_content_list.json"
    )
    if not content_list_path.exists():
        raise FileNotFoundError(
            f"MinerU output not found at {content_list_path}. "
            "Run the full pipeline first (python main.py) to generate it."
        )

    docs = load_document_mineru(str(content_list_path))
    print(f"  📄 Vanilla baseline: loaded {len(docs)} documents from MinerU cache")

    # Default LlamaIndex chunking (1024 tokens, 200 overlap)
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"  📦 Vanilla baseline: {len(nodes)} nodes (1024-token chunks)")

    # In-memory vector index
    index = VectorStoreIndex(nodes=nodes)

    # Simple retriever query engine (top_k=10, no reranking)
    retriever = index.as_retriever(similarity_top_k=10)
    engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)

    return engine


# ---------------------------------------------------------------------------
# Query execution helpers
# ---------------------------------------------------------------------------

def _run_full_pipeline_query(ctx: dict, query: str) -> dict:
    """Run query through the full pipeline. Returns {response_text, source_nodes, latency}."""
    from main import run_query

    result = run_query(ctx, query, verbose=False)
    return {
        "response_text": result["response_text"],
        "source_nodes": result["source_nodes"],
        "latency": result["latency_seconds"],
    }


def _run_vanilla_query(engine: Any, query: str) -> dict:
    """Run query through the vanilla baseline. Returns {response_text, source_nodes, latency}."""
    start = time.time()
    response = engine.query(query)
    latency = time.time() - start
    return {
        "response_text": str(response) if response else "",
        "source_nodes": response.source_nodes if response else [],
        "latency": latency,
    }


def _get_node_texts(source_nodes) -> list[str]:
    """Extract text from source nodes (handles both NodeWithScore and TextNode)."""
    texts = []
    for node in source_nodes:
        if hasattr(node, "node"):
            texts.append(node.node.get_content())
        elif hasattr(node, "get_content"):
            texts.append(node.get_content())
        elif hasattr(node, "text"):
            texts.append(node.text)
    return texts


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_single(
    query: str,
    reference_contexts: list[str],
    reference_answer: str,
    response_text: str,
    source_nodes: list,
    label: str,
) -> dict:
    """Compute all 3 metrics for a single query result."""
    node_texts = _get_node_texts(source_nodes)

    # 1. Context Recall
    cr = compute_context_recall(reference_contexts, node_texts)

    # 2. Faithfulness (answer vs retrieved context)
    context_concat = "\n\n---\n\n".join(node_texts) if node_texts else "(no context retrieved)"
    faith = evaluate_faithfulness(
        query=query, context=context_concat, answer=response_text
    )

    # 3. Correctness (answer vs gold reference)
    correct = evaluate_correctness(
        query=query, reference_answer=reference_answer, generated_answer=response_text
    )

    return {
        "context_recall": cr.score,
        "context_recall_matched": cr.matched,
        "context_recall_total": cr.total,
        "faithfulness": faith.score,
        "faithfulness_reasoning": faith.reasoning,
        "correctness": correct.score,
        "correctness_reasoning": correct.reasoning,
        "num_source_nodes": len(source_nodes),
        "response_length": len(response_text),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(limit: int | None = None, skip_baseline: bool = False, start: int = 1) -> dict:
    """
    Run the full benchmark and return the results dict.

    Args:
        limit: Maximum number of queries to evaluate (default: all).
        skip_baseline: If True, skip the vanilla baseline comparison.
        start: 1-based index of the first query to evaluate (default: 1).
    """
    print("=" * 70)
    print("  RAG Benchmark: Paul Graham Essay")
    print("=" * 70)

    # 1. Load dataset
    print("\n📥 Loading gold dataset …")
    all_examples = download_and_cache()
    total = len(all_examples)
    start_idx = max(0, start - 1)                       # convert 1-based to 0-based
    end_idx = (start_idx + limit) if limit else total
    examples = all_examples[start_idx:end_idx]
    if not examples:
        print(f"   ⚠️  No queries in range (start={start}, limit={limit}, total={total})")
        return {}
    print(f"   Queries {start_idx+1}–{start_idx+len(examples)} of {total} ({len(examples)} to evaluate)\n")

    # 2. Setup full pipeline
    print("🔧 Setting up full RAG pipeline …")
    from main import setup_pipeline
    ctx = setup_pipeline("paul_graham_essay")
    print("   ✅ Full pipeline ready\n")

    # 3. Setup vanilla baseline
    vanilla_engine = None
    if not skip_baseline:
        print("🔧 Setting up vanilla baseline …")
        vanilla_engine = _build_vanilla_baseline()
        print("   ✅ Vanilla baseline ready\n")

    # 4. Run queries and evaluate
    pipeline_results = []
    vanilla_results = []

    for i, ex in enumerate(examples):
        query = ex["query"]
        ref_ctx = ex["reference_contexts"]
        ref_ans = ex["reference_answer"]

        qnum = start_idx + i + 1   # 1-based global query number
        print(f"\n{'─' * 60}")
        print(f"  Query {qnum}/{total}: {query[:70]}{'…' if len(query) > 70 else ''}")
        print(f"{'─' * 60}")

        # --- Full pipeline ---
        print("  ▸ Full pipeline …", end=" ", flush=True)
        try:
            pipe_out = _run_full_pipeline_query(ctx, query)
            pipe_eval = evaluate_single(
                query, ref_ctx, ref_ans,
                pipe_out["response_text"],
                pipe_out["source_nodes"],
                "pipeline",
            )
            pipe_eval["latency"] = pipe_out["latency"]
            pipe_eval["query"] = query
            pipe_eval["response_text"] = pipe_out["response_text"]
            pipeline_results.append(pipe_eval)
            print(
                f"CR={pipe_eval['context_recall']:.2f}  "
                f"Faith={pipe_eval['faithfulness']}  "
                f"Correct={pipe_eval['correctness']}  "
                f"({pipe_out['latency']:.1f}s)"
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            pipeline_results.append({"query": query, "error": str(e)})

        # --- Vanilla baseline ---
        if vanilla_engine and not skip_baseline:
            print("  ▸ Vanilla baseline …", end=" ", flush=True)
            try:
                van_out = _run_vanilla_query(vanilla_engine, query)
                van_eval = evaluate_single(
                    query, ref_ctx, ref_ans,
                    van_out["response_text"],
                    van_out["source_nodes"],
                    "vanilla",
                )
                van_eval["latency"] = van_out["latency"]
                van_eval["query"] = query
                van_eval["response_text"] = van_out["response_text"]
                vanilla_results.append(van_eval)
                print(
                    f"CR={van_eval['context_recall']:.2f}  "
                    f"Faith={van_eval['faithfulness']}  "
                    f"Correct={van_eval['correctness']}  "
                    f"({van_out['latency']:.1f}s)"
                )
            except Exception as e:
                print(f"❌ Error: {e}")
                vanilla_results.append({"query": query, "error": str(e)})

    # 5. Aggregate and report
    report = _aggregate(pipeline_results, vanilla_results, skip_baseline, start_idx=start_idx)

    # 6. Print summary
    _print_summary(report)

    # 7. Save to disk
    _save_report(report)

    # Cleanup Milvus connection
    try:
        vector_store = ctx.get("vector_store")
        collection_name = ctx.get("collection_name_vector")
        if vector_store and hasattr(vector_store, "client"):
            vector_store.client.release_collection(collection_name=collection_name)
            vector_store.client.close()
    except Exception:
        pass

    return report


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else 0.0


def _aggregate(pipeline_results: list[dict], vanilla_results: list[dict], skip_baseline: bool, *, start_idx: int = 0) -> dict:
    """Aggregate per-query results into summary statistics."""
    def _summarise(results: list[dict]) -> dict:
        valid = [r for r in results if "error" not in r]
        errors = [r for r in results if "error" in r]
        return {
            "num_queries": len(results),
            "num_errors": len(errors),
            "context_recall_mean": _safe_mean([r["context_recall"] for r in valid]),
            "faithfulness_mean": _safe_mean([r["faithfulness"] for r in valid]),
            "correctness_mean": _safe_mean([r["correctness"] for r in valid]),
            "latency_mean": _safe_mean([r.get("latency", 0) for r in valid]),
        }

    # Build per-query list: each entry has pipeline + optional vanilla side-by-side
    per_query: list[dict] = []
    for i, pipe_r in enumerate(pipeline_results):
        entry: dict = {
            "query_number": start_idx + i + 1,
            "query": pipe_r.get("query", ""),
            "pipeline": pipe_r,
        }
        if not skip_baseline and i < len(vanilla_results):
            entry["vanilla_baseline"] = vanilla_results[i]
        per_query.append(entry)

    report: dict = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "pipeline": _summarise(pipeline_results),
        },
        "per_query": per_query,
    }
    if not skip_baseline:
        report["summary"]["vanilla_baseline"] = _summarise(vanilla_results)
    return report


def _print_summary(report: dict) -> None:
    """Print a formatted summary table."""
    summary = report["summary"]
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    header = f"{'Metric':<25} {'Full Pipeline':>15}"
    divider = f"{'─' * 25} {'─' * 15}"
    if "vanilla_baseline" in summary:
        header += f" {'Vanilla Baseline':>18} {'Δ':>8}"
        divider += f" {'─' * 18} {'─' * 8}"

    print(header)
    print(divider)

    pipe = summary["pipeline"]
    metrics = [
        ("Context Recall", "context_recall_mean", ".3f"),
        ("Faithfulness (1-5)", "faithfulness_mean", ".2f"),
        ("Correctness (1-5)", "correctness_mean", ".2f"),
        ("Avg Latency (s)", "latency_mean", ".1f"),
    ]

    for name, key, fmt in metrics:
        p_val = pipe[key]
        line = f"{name:<25} {p_val:>15{fmt}}"
        if "vanilla_baseline" in summary:
            v_val = summary["vanilla_baseline"][key]
            delta = p_val - v_val
            sign = "+" if delta >= 0 else ""
            line += f" {v_val:>18{fmt}} {sign}{delta:>7{fmt}}"
        print(line)

    pipe_errors = pipe["num_errors"]
    line = f"{'Errors':<25} {pipe_errors:>15d}"
    if "vanilla_baseline" in summary:
        v_errors = summary["vanilla_baseline"]["num_errors"]
        line += f" {v_errors:>18d}"
    print(line)

    print("=" * 70)


def _save_report(report: dict) -> None:
    """Save the full report as timestamped JSON + a Markdown companion.

    Filenames include evaluated query number(s), e.g.:
    - benchmark_q4_20260215_153130.json
    - benchmark_q4-6_20260215_160012.md
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build query number suffix from report payload
    qnums = [
        q.get("query_number")
        for q in report.get("per_query", [])
        if isinstance(q.get("query_number"), int)
    ]
    q_suffix = ""
    if qnums:
        q_unique = sorted(set(qnums))
        if len(q_unique) == 1:
            q_suffix = f"q{q_unique[0]}_"
        elif q_unique == list(range(q_unique[0], q_unique[-1] + 1)):
            q_suffix = f"q{q_unique[0]}-{q_unique[-1]}_"
        else:
            q_suffix = f"q{q_unique[0]}-{q_unique[-1]}_"

    # --- JSON (machine-readable) ---
    json_path = RESULTS_DIR / f"benchmark_{q_suffix}{ts}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # --- Markdown (human-readable, response_text renders correctly) ---
    md_path = RESULTS_DIR / f"benchmark_{q_suffix}{ts}.md"
    with open(md_path, "w") as f:
        f.write(_build_markdown_report(report))

    print(f"\n📊 Full report saved to {json_path}")
    print(f"📝 Markdown report saved to {md_path}")


def _build_markdown_report(report: dict) -> str:
    """Build a fully-renderable Markdown report from the results dict."""
    lines: list[str] = []
    _a = lines.append  # shorthand

    _a("# RAG Benchmark Report")
    _a("")
    _a(f"**Generated:** {report.get('timestamp', 'N/A')}")
    _a("")

    # --- Summary table ---
    summary = report.get("summary", {})
    sections = [("pipeline", "Full Pipeline")]
    if "vanilla_baseline" in summary:
        sections.append(("vanilla_baseline", "Vanilla Baseline"))

    _a("## Summary")
    _a("")
    _a("| Metric | " + " | ".join(label for _, label in sections) + " |")
    _a("| --- | " + " | ".join("---" for _ in sections) + " |")
    for metric, fmt in [
        ("context_recall_mean", ".3f"),
        ("faithfulness_mean", ".2f"),
        ("correctness_mean", ".2f"),
        ("latency_mean", ".1f"),
        ("num_queries", "d"),
        ("num_errors", "d"),
    ]:
        nice = metric.replace("_mean", "").replace("_", " ").title()
        if "mean" in metric:
            nice += " (mean)"
        vals = []
        for key, _ in sections:
            v = summary.get(key, {}).get(metric, "—")
            try:
                vals.append(f"{v:{fmt}}")
            except (ValueError, TypeError):
                vals.append(str(v))
        _a(f"| {nice} | " + " | ".join(vals) + " |")
    _a("")

    # --- Per-query details (pipeline + vanilla grouped per query) ---
    has_vanilla = "vanilla_baseline" in summary
    per_query = report.get("per_query", [])

    for entry in per_query:
        qnum = entry.get("query_number", "?")
        query_text = entry.get("query", "N/A")

        _a(f"## Query {qnum}")
        _a("")
        _a(f"> {query_text}")
        _a("")

        for variant_key, variant_label in [("pipeline", "Full Pipeline"), ("vanilla_baseline", "Vanilla Baseline")]:
            q = entry.get(variant_key)
            if q is None:
                continue

            _a(f"### {variant_label}")
            _a("")

            if "error" in q:
                _a(f"**Error:** {q['error']}")
                _a("")
                continue

            # Scores table
            _a("| Context Recall | Faithfulness | Correctness | Latency (s) | Source Nodes | Response Length |")
            _a("| --- | --- | --- | --- | --- | --- |")
            _a(
                f"| {q.get('context_recall', 0):.3f} "
                f"| {q.get('faithfulness', '—')} "
                f"| {q.get('correctness', '—')} "
                f"| {q.get('latency', 0):.1f} "
                f"| {q.get('num_source_nodes', '—')} "
                f"| {q.get('response_length', '—')} |"
            )
            _a("")

            # Reasoning
            if q.get("faithfulness_reasoning"):
                _a(f"**Faithfulness reasoning:** {q['faithfulness_reasoning']}")
                _a("")
            if q.get("correctness_reasoning"):
                _a(f"**Correctness reasoning:** {q['correctness_reasoning']}")
                _a("")

            # Response text — written verbatim so Markdown headers/lists render
            _a("<details>")
            _a(f"<summary>Response text ({q.get('response_length', '?')} chars)</summary>")
            _a("")
            _a(q.get("response_text", ""))
            _a("")
            _a("</details>")
            _a("")

    _a("---")
    _a("*Report generated by `benchmark.run_benchmark`*")
    _a("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG benchmark evaluation")
    parser.add_argument(
        "--start", type=int, default=1,
        help="1-based index of the first query to run (default: 1)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of queries to evaluate from --start (default: all)"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the vanilla baseline comparison"
    )
    args = parser.parse_args()

    run_benchmark(limit=args.limit, skip_baseline=args.skip_baseline, start=args.start)
