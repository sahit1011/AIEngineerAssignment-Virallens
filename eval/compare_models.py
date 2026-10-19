"""Compare Groq models on the classification roles: how often each returns invalid output, and how
accurate it is, on the evaluation questions.

    uv run python eval/compare_models.py [--trials 2] [--out eval/model_comparison.json]

Each model answers the same strict-schema calls: routing the 15 questions, grading their retrieved
passages, and verifying answers twice over — 15 genuine answers (drafted once by the answer model
from passages on the expected pages) and the same answers paired with another question's passages,
which a verifier must reject. Calls bypass the retry/fallback policy, so invalid output is counted.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import groq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from virallens.common.eval_set import EvalQuestion, load_questions
from virallens.rag import prompts
from virallens.rag.agents import Answer, Grade, Route, Verdict
from virallens.rag.citations import citation_key, format_context, normalise_citations
from virallens.rag.config import Settings, load_settings
from virallens.rag.llm import reasoning_effort
from virallens.rag.retrieval import HybridIndex, corpus_fingerprint

CANDIDATES = ("qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b")
# The evidence an answer is drafted and verified from: up to 3 passages on the expected pages.
EVIDENCE_PASSAGES = 3
# The mismatched answer for question i is the one drafted for question i + OFFSET.
MISMATCH_OFFSET = 7


def _messages(system: str, human: str) -> list[BaseMessage]:
    return [SystemMessage(system), HumanMessage(human)]


def _model(settings: Settings, model: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=0,
        reasoning_effort=reasoning_effort(model),
        max_retries=4,
    )


def _error_code(error: groq.APIStatusError) -> str:
    body = error.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return f"{type(error).__name__}:{body['error'].get('code')}"
    return type(error).__name__


def compare(settings: Settings, trials: int) -> dict[str, Any]:
    index = HybridIndex.load(
        settings.index_dir, settings.fastembed_cache_dir, corpus_fingerprint(settings.data_dir)
    )
    questions = load_questions(settings.eval_questions)
    candidates = {q.id: [hit.chunk for hit in index.search(q.question, 6)] for q in questions}
    evidence = {
        q.id: [h.chunk for h in index.search(q.question, 20) if q.is_hit(h.chunk)][
            :EVIDENCE_PASSAGES
        ]
        for q in questions
    }
    drafter = _model(settings, settings.groq_answer_model).with_structured_output(
        Answer, method="json_schema", strict=True
    )
    answers: dict[str, str] = {}
    for q in questions:
        prompt = f"Passages:\n\n{format_context(evidence[q.id])}\n\nQuestion: {q.question}"
        draft = cast(Answer, drafter.invoke(_messages(prompts.ANSWERER, prompt)))
        answers[q.id] = normalise_citations(draft.answer)
    calls: list[dict[str, Any]] = []
    for model in CANDIDATES:
        llm = _model(settings, model)
        for trial in range(trials):
            for i, q in enumerate(questions):
                mismatched = answers[questions[(i + MISMATCH_OFFSET) % len(questions)].id]
                verify = f"Passages:\n\n{format_context(evidence[q.id])}\n\nQuestion: {q.question}"
                outcome, route = _call(llm, Route, prompts.ROUTER, q.question)
                routed = route is not None and route.route == q.route
                calls.append(_row(model, trial, q, "router", outcome, routed))
                grading = (
                    f"Question: {q.question}\n\nPassages:\n\n{format_context(candidates[q.id])}"
                )
                outcome, grade = _call(llm, Grade, prompts.GRADER, grading)
                kept = grade is not None and _grader_keeps_expected(q, candidates[q.id], grade)
                calls.append(_row(model, trial, q, "grader", outcome, kept))
                genuine = f"{verify}\n\nAnswer:\n{answers[q.id]}"
                outcome, verdict = _call(llm, Verdict, prompts.VERIFIER, genuine)
                accepted = verdict is not None and verdict.supported
                calls.append(_row(model, trial, q, "verifier_genuine", outcome, accepted))
                wrong = f"{verify}\n\nAnswer:\n{mismatched}"
                outcome, verdict = _call(llm, Verdict, prompts.VERIFIER, wrong)
                rejected = verdict is not None and not verdict.supported
                calls.append(_row(model, trial, q, "verifier_mismatched", outcome, rejected))
    return {"answer_model": settings.groq_answer_model, "trials": trials, "calls": calls}


def _row(
    model: str, trial: int, question: EvalQuestion, role: str, outcome: str, correct: bool
) -> dict[str, Any]:
    return {
        "model": model,
        "trial": trial,
        "question": question.id,
        "role": role,
        "outcome": outcome,
        "correct": correct,
    }


def _call(llm: ChatGroq, schema: type, system: str, human: str) -> tuple[str, Any]:
    """One strict-schema call: ("ok", parsed) or (error code, None) for invalid model output."""
    try:
        structured = llm.with_structured_output(schema, method="json_schema", strict=True)
        return "ok", structured.invoke(_messages(system, human))
    except groq.BadRequestError as error:
        return _error_code(error), None


def _grader_keeps_expected(question: EvalQuestion, candidates: list, grade: Any) -> bool:
    kept = set(grade.relevant_keys)
    return any(question.is_hit(c) for c in candidates if citation_key(c) in kept)


def summarise(result: dict[str, Any]) -> None:
    for model in CANDIDATES:
        rows = [c for c in result["calls"] if c["model"] == model]
        invalid = Counter(c["outcome"] for c in rows if c["outcome"] != "ok")
        print(f"{model}: invalid outputs {sum(invalid.values())}/{len(rows)} {dict(invalid)}")
        for role in ("router", "grader", "verifier_genuine", "verifier_mismatched"):
            role_rows = [c for c in rows if c["role"] == role]
            print(f"  {role:20s} correct {sum(c['correct'] for c in role_rows)}/{len(role_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("eval/model_comparison.json"))
    args = parser.parse_args()
    result = compare(load_settings(), args.trials)
    args.out.write_text(json.dumps(result, indent=1))
    summarise(result)


if __name__ == "__main__":
    main()
