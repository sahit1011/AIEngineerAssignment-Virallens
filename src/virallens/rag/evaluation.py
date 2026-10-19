"""Evaluation of retrieval and of the full pipeline against the hand-written questions."""

from collections.abc import Iterable
from dataclasses import dataclass

import groq
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.exceptions import OutputParserException
from openrouter.errors import OpenRouterError

from virallens.common.eval_set import EvalQuestion
from virallens.rag.citations import citation_key
from virallens.rag.graph import RAGPipeline, Status
from virallens.rag.retrieval import HybridIndex, Retriever

# Failures of a single question (provider error, malformed model output, an agent over its call
# budget) are recorded rather than ending the run; `main.py eval-e2e` exits non-zero if any occur.
QUESTION_FAILURES = (
    groq.APIError,
    OpenRouterError,
    OutputParserException,
    ModelCallLimitExceededError,
)


@dataclass(frozen=True)
class RetrievalReport:
    retrievers: tuple[Retriever, ...]
    k: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    misses: list[str]


def evaluate_retrieval(
    index: HybridIndex, questions: list[EvalQuestion], k: int, retrievers: tuple[Retriever, ...]
) -> RetrievalReport:
    reciprocal_ranks = []
    for question in questions:
        hits = index.search(question.question, k, retrievers=retrievers)
        ranks = [rank for rank, hit in enumerate(hits, start=1) if question.is_hit(hit.chunk)]
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)
    n = len(questions)
    return RetrievalReport(
        retrievers=retrievers,
        k=k,
        hit_at_1=sum(rr == 1.0 for rr in reciprocal_ranks) / n,
        hit_at_k=sum(rr > 0 for rr in reciprocal_ranks) / n,
        mrr=sum(reciprocal_ranks) / n,
        misses=[q.id for q, rr in zip(questions, reciprocal_ranks, strict=True) if rr == 0],
    )


@dataclass(frozen=True)
class EndToEndRow:
    id: str
    expected_route: str
    route: str
    status: Status
    cites_expected_source: bool
    # The passages the answer cites, as "<key> <pages>", so a result can be checked against them.
    cited: list[str]
    total_tokens: int


@dataclass(frozen=True)
class EndToEndFailure:
    id: str
    error: str


@dataclass(frozen=True)
class EndToEndReport:
    rows: list[EndToEndRow]
    failures: list[EndToEndFailure]

    @property
    def route_accuracy(self) -> float:
        return _share(row.route == row.expected_route for row in self.rows)

    @property
    def verified_rate(self) -> float:
        return _share(row.status == "verified" for row in self.rows)

    @property
    def cites_expected_rate(self) -> float:
        return _share(row.cites_expected_source for row in self.rows)

    @property
    def total_tokens(self) -> int:
        return sum(row.total_tokens for row in self.rows)


def evaluate_end_to_end(pipeline: RAGPipeline, questions: list[EvalQuestion]) -> EndToEndReport:
    rows, failures = [], []
    for question in questions:
        try:
            result = pipeline.ask(question.question)
        except QUESTION_FAILURES as error:
            failures.append(EndToEndFailure(question.id, f"{type(error).__name__}: {error}"))
            continue
        rows.append(
            EndToEndRow(
                id=question.id,
                expected_route=question.route,
                route=result.route,
                status=result.status,
                cites_expected_source=any(question.is_hit(c) for c in result.sources),
                cited=[f"{citation_key(c)} {c.pages}" for c in result.sources],
                total_tokens=sum(u["total_tokens"] for u in result.token_usage.values()),
            )
        )
    return EndToEndReport(rows=rows, failures=failures)


def _share(flags: Iterable[bool]) -> float:
    values = list(flags)
    if not values:
        raise ValueError("no answered questions to compute a rate over")
    return sum(values) / len(values)
