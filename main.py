"""Command line for the multi-agent RAG over the filings in data/ (Q1).

python main.py ingest                         # parse, chunk and index data/*.pdf (no API key)
python main.py ask "Are pure omissions actionable under Rule 10b-5(b)?"
python main.py eval-retrieval                 # retrieval metrics (no API key)
python main.py eval-e2e [--ids ID ...] [--out results.json]
"""

import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from virallens.common.catalog import CATALOG
from virallens.common.corpus import load_corpus
from virallens.common.eval_set import load_questions
from virallens.common.settings import Paths
from virallens.rag.citations import citation_key
from virallens.rag.config import Settings, load_settings
from virallens.rag.evaluation import evaluate_end_to_end, evaluate_retrieval
from virallens.rag.graph import RETRIEVE_K, RAGPipeline, RAGResult
from virallens.rag.retrieval import HYBRID, HybridIndex, Retriever, corpus_fingerprint

STATUS_LABELS = {
    "verified": "verified",
    "unverified": "NOT VERIFIED",
    "no_evidence": "no supporting passages found",
}


def ingest(paths: Paths) -> None:
    chunks = load_corpus(paths.data_dir)
    index = HybridIndex.build(chunks, paths.fastembed_cache_dir, corpus_fingerprint(paths.data_dir))
    index.save(paths.index_dir)
    print(f"Indexed {len(chunks)} chunks from {len(CATALOG)} filings into {paths.index_dir}")


def ask(settings: Settings, question: str) -> RAGResult:
    result = RAGPipeline.from_settings(settings).ask(question)
    print(f"\n{result.answer}\n")
    print(f"[{STATUS_LABELS[result.status]} | route: {result.route}]")
    for claim in result.unsupported_claims:
        print(f"  unsupported: {claim}")
    print("\nSources:")
    for chunk in result.sources:
        print(f"  [{citation_key(chunk)}] {CATALOG[chunk.doc_id].title}, {chunk.pages}")
        print(f"      {chunk.section}")
    print("\nTrace:")
    for step in result.trace:
        print(f"  {step}")
    print("\nTokens:", {model: u["total_tokens"] for model, u in result.token_usage.items()})
    return result


def eval_retrieval(paths: Paths) -> None:
    index = HybridIndex.load(
        paths.index_dir, paths.fastembed_cache_dir, corpus_fingerprint(paths.data_dir)
    )
    questions = load_questions(paths.eval_questions)
    print(f"{len(questions)} questions, k={RETRIEVE_K}")
    retriever_sets: tuple[tuple[Retriever, ...], ...] = (("bm25",), ("dense",), HYBRID)
    for retrievers in retriever_sets:
        report = evaluate_retrieval(index, questions, RETRIEVE_K, retrievers)
        print(
            f"  {'+'.join(retrievers):11s} hit@1={report.hit_at_1:.2f} "
            f"hit@{report.k}={report.hit_at_k:.2f} MRR={report.mrr:.3f} misses={report.misses}"
        )


def eval_e2e(settings: Settings, ids: list[str] | None, out: Path | None) -> None:
    questions = load_questions(settings.eval_questions)
    if ids:
        unknown = set(ids) - {q.id for q in questions}
        if unknown:
            raise SystemExit(f"unknown question ids: {sorted(unknown)}")
        questions = [q for q in questions if q.id in ids]
    report = evaluate_end_to_end(RAGPipeline.from_settings(settings), questions)
    print(f"{'id':24s} {'route':26s} {'status':12s} cites-expected  tokens")
    for row in report.rows:
        route = f"{row.route} (expected {row.expected_route})"
        print(
            f"{row.id:24s} {route:26s} {row.status:12s} "
            f"{row.cites_expected_source!s:14s}  {row.total_tokens}"
        )
    if report.rows:
        print(
            f"\n{len(report.rows)} answered | route accuracy {report.route_accuracy:.2f}"
            f" | verified {report.verified_rate:.2f}"
            f" | cites expected source {report.cites_expected_rate:.2f}"
            f" | tokens {report.total_tokens}"
        )
    for failure in report.failures:
        print(f"FAILED {failure.id}: {failure.error}")
    if out is not None:
        run = {
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "models": {
                "classifier": settings.groq_classifier_model,
                "answer": settings.groq_answer_model,
                "analyst": settings.groq_analyst_model,
            },
            "fallback": settings.openrouter_model if settings.openrouter_api_key else None,
        }
        out.write_text(json.dumps({"run": run, **asdict(report)}, indent=2))
        print(f"Results written to {out}")
    if report.failures:
        raise SystemExit(f"{len(report.failures)} of {len(questions)} questions failed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log every agent step")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ingest", help="parse, chunk and index the filings")
    ask_parser = commands.add_parser("ask", help="answer a question")
    ask_parser.add_argument("question")
    commands.add_parser("eval-retrieval", help="retrieval metrics on the eval questions")
    e2e_parser = commands.add_parser("eval-e2e", help="run the full pipeline on the eval questions")
    e2e_parser.add_argument("--ids", nargs="+", help="only these question ids")
    e2e_parser.add_argument("--out", type=Path, help="also write the results as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "ingest":
        ingest(Paths())
    elif args.command == "eval-retrieval":
        eval_retrieval(Paths())
    elif args.command == "ask":
        ask(load_settings(), args.question)
    elif args.command == "eval-e2e":
        eval_e2e(load_settings(), args.ids, args.out)


if __name__ == "__main__":
    main()
