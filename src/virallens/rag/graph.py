"""The LangGraph orchestration of the RAG agents (diagram: README.md, "LangGraph flow").

A direct question is retrieved, graded (with one query rewrite if nothing is relevant), answered and
verified, with one revision if the verifier finds unsupported claims. A research question goes to
the DeepAgents planner; its answer and the passages its analysts retrieved go through the same
verifier, and the same answerer revises it.
"""

import logging
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from virallens.common.chunking import Chunk
from virallens.rag.agents import Agents
from virallens.rag.citations import citation_key, cited_keys
from virallens.rag.config import Settings
from virallens.rag.llm import Models
from virallens.rag.research import Researcher
from virallens.rag.retrieval import HybridIndex, corpus_fingerprint

logger = logging.getLogger(__name__)

# Passages retrieved per query: 6 chunks (~1.5k tokens) keep a direct question's four calls inside a
# minute of Groq's 8000 tokens/min.
RETRIEVE_K = 6
# One rewrite recovers a query phrased unlike the filings; more rarely helps and spends quota.
MAX_REWRITES = 1
# One revision fixes a mis-cited claim; an answer still unsupported after it is returned flagged.
MAX_REVISIONS = 1
# A revision sees the draft's cited passages first, then the rest, up to this many (~3k tokens): the
# research path can collect dozens, more than one call's share of the token quota.
MAX_REVISION_PASSAGES = 2 * RETRIEVE_K
NO_EVIDENCE_ANSWER = "The filings do not contain passages that answer this question."
UNNAMED_CLAIM = "The verifier judged the answer unsupported without naming a claim."


class RAGState(TypedDict):
    question: str
    route: str
    query: str
    rewrites: int
    candidates: list[Chunk]
    passages: list[Chunk]
    answer: str
    unsupported_claims: list[str]
    verified: bool
    revisions: int
    trace: Annotated[list[str], operator.add]


Status = Literal["verified", "unverified", "no_evidence"]


@dataclass(frozen=True)
class RAGResult:
    question: str
    route: str
    answer: str
    status: Status
    unsupported_claims: list[str]
    sources: list[Chunk]
    trace: list[str]
    token_usage: dict[str, Any]

    @property
    def verified(self) -> bool:
        return self.status == "verified"


class RAGPipeline:
    def __init__(self, agents: Agents, researcher: Researcher, index: HybridIndex):
        self._agents = agents
        self._researcher = researcher
        self._index = index
        self.graph = self._build()

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        models = Models(settings)
        index = HybridIndex.load(
            settings.index_dir, settings.fastembed_cache_dir, corpus_fingerprint(settings.data_dir)
        )
        return cls(Agents(models), Researcher(models, index), index)

    def ask(self, question: str) -> RAGResult:
        initial: RAGState = {
            "question": question,
            "route": "",
            "query": question,
            "rewrites": 0,
            "candidates": [],
            "passages": [],
            "answer": "",
            "unsupported_claims": [],
            "verified": False,
            "revisions": 0,
            "trace": [],
        }
        with get_usage_metadata_callback() as usage:
            state = self.graph.invoke(initial)
        cited = set(cited_keys(state["answer"]))
        status: Status = (
            "no_evidence"
            if not state["passages"]
            else "verified"
            if state["verified"]
            else "unverified"
        )
        return RAGResult(
            question=question,
            route=state["route"],
            answer=state["answer"],
            status=status,
            unsupported_claims=state["unsupported_claims"],
            sources=[c for c in state["passages"] if citation_key(c) in cited],
            trace=state["trace"],
            token_usage=dict(usage.usage_metadata),
        )

    def _build(self) -> CompiledStateGraph:
        graph = StateGraph(RAGState)
        graph.add_node("route", self._route)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("research", self._research)
        graph.add_node("answer", self._answer)
        graph.add_node("verify", self._verify)
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route", lambda s: "research" if s["route"] == "research" else "retrieve"
        )
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", _after_grade)
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("research", "verify")
        graph.add_conditional_edges("answer", lambda s: "verify" if s["passages"] else END)
        graph.add_conditional_edges("verify", _after_verify)
        return graph.compile()

    def _route(self, state: RAGState) -> dict[str, Any]:
        decision = self._agents.route(state["question"])
        return _step(f"router: {decision.route} ({decision.reason})", route=decision.route)

    def _retrieve(self, state: RAGState) -> dict[str, Any]:
        hits = self._index.search(state["query"], RETRIEVE_K)
        keys = ", ".join(citation_key(hit.chunk) for hit in hits)
        return _step(
            f"retriever: {state['query']!r} -> {keys}", candidates=[hit.chunk for hit in hits]
        )

    def _grade(self, state: RAGState) -> dict[str, Any]:
        grade = self._agents.grade(state["question"], state["candidates"])
        keys = set(grade.relevant_keys)
        passages = [c for c in state["candidates"] if citation_key(c) in keys]
        unknown = keys - {citation_key(c) for c in state["candidates"]}
        detail = f"grader: {len(passages)}/{len(state['candidates'])} relevant"
        if unknown:
            detail += f" (ignored keys not retrieved: {sorted(unknown)})"
        return _step(detail, passages=passages)

    def _rewrite(self, state: RAGState) -> dict[str, Any]:
        rewrite = self._agents.rewrite(state["question"], state["query"])
        return _step(
            f"rewriter: {rewrite.query!r}", query=rewrite.query, rewrites=state["rewrites"] + 1
        )

    def _research(self, state: RAGState) -> dict[str, Any]:
        result = self._researcher.run(state["question"])
        return _step(
            f"research: deep agent read {len(result.passages)} passages",
            answer=result.answer,
            passages=result.passages,
        )

    def _answer(self, state: RAGState) -> dict[str, Any]:
        if not state["passages"]:
            # Claims about a discarded answer would describe text the user never sees.
            return _step(
                "answerer: no relevant passages",
                answer=NO_EVIDENCE_ANSWER,
                unsupported_claims=[],
                verified=False,
            )
        claims = state["unsupported_claims"]
        if not claims:
            answer = self._agents.answer(state["question"], state["passages"])
            return _step("answerer: drafted answer", answer=answer.answer)
        evidence = _revision_evidence(state["answer"], state["passages"])
        answer = self._agents.revise(state["question"], evidence, state["answer"], claims)
        return _step(
            "answerer: revised answer", answer=answer.answer, revisions=state["revisions"] + 1
        )

    def _verify(self, state: RAGState) -> dict[str, Any]:
        claims = _citation_problems(state["answer"], state["passages"])
        if not claims:
            # The code check guarantees every cited key is a retrieved passage, so the model only
            # needs those; the research path can retrieve far more than an answer cites.
            cited = set(cited_keys(state["answer"]))
            evidence = [c for c in state["passages"] if citation_key(c) in cited]
            verdict = self._agents.verify(state["question"], state["answer"], evidence)
            claims = verdict.unsupported_claims
            if not verdict.supported and not claims:
                claims = [UNNAMED_CLAIM]
        detail = "verifier: supported" if not claims else f"verifier: {len(claims)} unsupported"
        return _step(detail, verified=not claims, unsupported_claims=claims)


def _revision_evidence(draft: str, passages: list[Chunk]) -> list[Chunk]:
    cited = set(cited_keys(draft))
    ordered = sorted(passages, key=lambda c: citation_key(c) not in cited)
    return ordered[:MAX_REVISION_PASSAGES]


def _citation_problems(answer: str, passages: list[Chunk]) -> list[str]:
    """Checks that need no model: the answer cites something, and only passages it was given."""
    cited = cited_keys(answer)
    if not cited:
        return ["The answer cites no passages."]
    known = {citation_key(c) for c in passages}
    return [
        f"Cites [{key}], which is not among the retrieved passages."
        for key in cited
        if key not in known
    ]


def _after_grade(state: RAGState) -> Literal["answer", "rewrite"]:
    if state["passages"] or state["rewrites"] >= MAX_REWRITES:
        return "answer"
    return "rewrite"


def _after_verify(state: RAGState) -> str:
    if state["verified"] or state["revisions"] >= MAX_REVISIONS:
        return END
    return "answer"


def _step(detail: str, **updates: Any) -> dict[str, Any]:
    logger.info(detail)
    return {**updates, "trace": [detail]}
