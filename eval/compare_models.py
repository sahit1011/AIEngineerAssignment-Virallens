"""Compare Groq models on the classification roles: how often each returns invalid output, and how
accurate it is on the evaluation questions.

    uv run python eval/compare_models.py [--trials 2] [--out eval/model_comparison.json]

Two trials take on the order of 100k tokens per candidate model, half of Groq's 200k/day cap, so
run it on a day those models are otherwise idle (not on the same day as a full `eval-e2e`).

Per candidate model, trial and question:

- router: the route, compared with the question's label;
- grader: the passages kept from the retrieved candidates, scored against the candidates on
  expected pages (recall where there are any, precision over everything kept);
- verifier, genuine: an answer drafted once from passages on the expected pages, shown with the
  passages it cites, as in production; it should be judged supported;
- verifier, swapped: another question's answer with its citations rewritten to this question's
  passages, so the citations are valid but the claims are not; it should be judged unsupported.

Candidate calls bypass the call policy, so invalid output is counted rather than retried; drafting
runs under it (one retry of invalid output, no fallback). Results are written after each model.
"""

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import groq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

from virallens.common.chunking import Chunk
from virallens.common.eval_set import EvalQuestion, load_questions
from virallens.rag import prompts
from virallens.rag.agents import Answer, Grade, Route, Verdict
from virallens.rag.citations import citation_key, cited_keys, format_context, normalise_citations
from virallens.rag.config import Settings, load_settings
from virallens.rag.graph import RETRIEVE_K
from virallens.rag.llm import (
    GROQ_MAX_RETRIES,
    call_with_policy,
    error_code,
    is_invalid_output,
    reasoning_effort,
)
from virallens.rag.retrieval import CANDIDATES_PER_RETRIEVER, HybridIndex, corpus_fingerprint

CANDIDATES = ("qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b")
# An answer is drafted and verified from up to 3 passages on the expected pages.
EVIDENCE_PASSAGES = 3
# The swapped answer for question i is the one drafted for question i + SWAP_OFFSET.
SWAP_OFFSET = 7


def compare(settings: Settings, trials: int, out: Path) -> dict[str, Any]:
    index = HybridIndex.load(
        settings.index_dir, settings.fastembed_cache_dir, corpus_fingerprint(settings.data_dir)
    )
    questions = load_questions(settings.eval_questions)
    candidates = {q.id: [h.chunk for h in index.search(q.question, RETRIEVE_K)] for q in questions}
    evidence = {
        q.id: [
            h.chunk for h in index.search(q.question, CANDIDATES_PER_RETRIEVER) if q.is_hit(h.chunk)
        ][:EVIDENCE_PASSAGES]
        for q in questions
    }
    answers = _draft_answers(settings, questions, evidence)
    result: dict[str, Any] = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "index_fingerprint": index.fingerprint,
        "answer_model": settings.groq_answer_model,
        "trials": trials,
        "answers": answers,
        "calls": [],
    }
    for model in CANDIDATES:
        llm = _model(settings, model)
        for trial in range(trials):
            for i, q in enumerate(questions):
                other = questions[(i + SWAP_OFFSET) % len(questions)]
                result["calls"] += _check_question(
                    llm,
                    {"model": model, "trial": trial, "question": q.id},
                    q,
                    candidates[q.id],
                    evidence[q.id],
                    answers[q.id],
                    answers[other.id],
                )
        out.write_text(json.dumps(result, indent=2))
    return result


def _check_question(
    llm: ChatGroq,
    context: dict[str, Any],
    question: EvalQuestion,
    candidates: list[Chunk],
    evidence: list[Chunk],
    answer: str,
    other_answer: str,
) -> list[dict[str, Any]]:
    """The four scored calls for one question, each row tagged with ``context``."""
    rows = []
    outcome, route = _call(llm, Route, prompts.ROUTER, question.question)
    routed = route is not None and route.route == question.route
    rows.append({**context, "role": "router", "outcome": outcome, "correct": routed})

    grading = f"Question: {question.question}\n\nPassages:\n\n{format_context(candidates)}"
    outcome, grade = _call(llm, Grade, prompts.GRADER, grading)
    kept = [] if grade is None else sorted(set(grade.relevant_keys))
    expected = [citation_key(c) for c in candidates if question.is_hit(c)]
    rows.append(
        {**context, "role": "grader", "outcome": outcome, "kept": kept, "expected": expected}
    )

    keys = [citation_key(c) for c in evidence]
    for role, text, supported in (
        ("verifier_genuine", answer, True),
        ("verifier_swapped", swap_citations(other_answer, keys), False),
    ):
        cited = [c for c in evidence if citation_key(c) in set(cited_keys(text))]
        prompt = (
            f"Passages:\n\n{format_context(cited)}\n\n"
            f"Question: {question.question}\n\nAnswer:\n{text}"
        )
        outcome, verdict = _call(llm, Verdict, prompts.VERIFIER, prompt)
        judged = verdict is not None and verdict.supported == supported
        rows.append({**context, "role": role, "outcome": outcome, "correct": judged})
    return rows


def swap_citations(answer: str, keys: list[str]) -> str:
    """Point every citation in ``answer`` at one of ``keys``, in turn."""
    for i, key in enumerate(dict.fromkeys(cited_keys(answer))):
        within_brackets = rf"(?<=[\[;,\s]){re.escape(key)}(?=[\];,])"
        answer = re.sub(within_brackets, keys[i % len(keys)], answer)
    return answer


def _draft_answers(
    settings: Settings, questions: list[EvalQuestion], evidence: dict[str, list[Chunk]]
) -> dict[str, str]:
    drafter = _model(settings, settings.groq_answer_model).with_structured_output(
        Answer, method="json_schema", strict=True
    )
    answers = {}
    for q in questions:
        prompt = f"Passages:\n\n{format_context(evidence[q.id])}\n\nQuestion: {q.question}"
        messages = _messages(prompts.ANSWERER, prompt)
        draft = cast(Answer, call_with_policy(lambda m=messages: drafter.invoke(m), None))
        answers[q.id] = normalise_citations(draft.answer)
    return answers


def _call(llm: ChatGroq, schema: type[BaseModel], system: str, human: str) -> tuple[str, Any]:
    """One strict-schema call: ("ok", parsed), or (error code, None) for invalid model output."""
    structured = llm.with_structured_output(schema, method="json_schema", strict=True)
    try:
        return "ok", structured.invoke(_messages(system, human))
    except groq.BadRequestError as error:
        if not is_invalid_output(error):
            raise
        return str(error_code(error)), None


def _model(settings: Settings, model: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=0,
        reasoning_effort=reasoning_effort(model),
        max_retries=GROQ_MAX_RETRIES,
    )


def _messages(system: str, human: str) -> list[BaseMessage]:
    return [SystemMessage(system), HumanMessage(human)]


def summarise(result: dict[str, Any]) -> None:
    for model in CANDIDATES:
        rows = [c for c in result["calls"] if c["model"] == model]
        if not rows:
            continue
        invalid = Counter(c["outcome"] for c in rows if c["outcome"] != "ok")
        print(f"{model}: invalid outputs {sum(invalid.values())}/{len(rows)} {dict(invalid)}")
        for role in ("router", "verifier_genuine", "verifier_swapped"):
            scored = [c for c in rows if c["role"] == role]
            print(f"  {role:18s} correct {sum(c['correct'] for c in scored)}/{len(scored)}")
        grades = [c for c in rows if c["role"] == "grader" and c["outcome"] == "ok"]
        with_expected = [c for c in grades if c["expected"]]
        recall = sum(bool(set(c["kept"]) & set(c["expected"])) for c in with_expected)
        kept = sum(len(c["kept"]) for c in grades)
        precise = sum(len(set(c["kept"]) & set(c["expected"])) for c in grades)
        print(
            f"  grader keeps an expected passage {recall}/{len(with_expected)}; "
            f"{precise}/{kept} kept passages are on expected pages"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("eval/model_comparison.json"))
    args = parser.parse_args()
    summarise(compare(load_settings(), args.trials, args.out))


if __name__ == "__main__":
    main()
