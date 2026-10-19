"""DeepAgents research branch for questions that span several filings.

A planner breaks the question into sub-questions with ``write_todos`` and delegates each to a
filing-analyst sub-agent through the ``task`` tool. The analyst is the planner's only delegate and
the only agent that can search, so every passage it retrieves is recorded and the graph's verifier
checks the answer against exactly those passages.
"""

from dataclasses import dataclass
from typing import Any

from deepagents import FilesystemMiddleware, SubAgent, create_deep_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
)
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, field_validator
from pydantic.config import JsonDict

from virallens.common.catalog import BY_SHORT_NAME
from virallens.common.chunking import Chunk
from virallens.rag import prompts
from virallens.rag.citations import citation_key, format_context, normalise_citations
from virallens.rag.llm import Models
from virallens.rag.retrieval import HybridIndex

# Passages per search: 4 (~1k tokens) keeps an analyst's 2-3 searches inside a minute of Groq quota.
SEARCH_K = 4
# Model-call budgets that stop a looping agent. Planner: a plan, up to 4 delegations each followed
# by a todo update, and the answer. Analyst: up to 3 searches and a report. Both plus margin.
PLANNER_MODEL_CALLS = 14
ANALYST_MODEL_CALLS = 6
# create_deep_agent adds its own "general-purpose" sub-agent (no call limit, free to answer from
# model memory) unless a spec already uses that name; the analyst takes it to be the only one.
ANALYST_NAME = "general-purpose"


# Shown to the model as the allowed values of `document`.
_FILING_CHOICES: JsonDict = {"enum": [*sorted(BY_SHORT_NAME)]}


class _SearchArgs(BaseModel):
    query: str = Field(description="Search query using the filings' own legal terms")
    document: str | None = Field(
        default=None,
        description="Restrict the search to one filing; omit to search all",
        json_schema_extra=_FILING_CHOICES,
    )

    @field_validator("document")
    @classmethod
    def _known_filing(cls, document: str | None) -> str | None:
        if document is not None and document not in BY_SHORT_NAME:
            raise ValueError(
                f"unknown filing {document!r}; expected one of {sorted(BY_SHORT_NAME)}"
            )
        return document


@dataclass
class ResearchResult:
    answer: str
    passages: list[Chunk]


class Researcher:
    def __init__(self, models: Models, index: HybridIndex):
        self._models = models
        self._index = index

    def run(self, question: str) -> ResearchResult:
        seen: dict[str, Chunk] = {}
        state = self.agent(seen).invoke({"messages": [HumanMessage(question)]})
        answer = normalise_citations(state["messages"][-1].text)
        return ResearchResult(answer=answer, passages=list(seen.values()))

    def agent(self, seen: dict[str, Chunk]) -> CompiledStateGraph:
        """The planner and its analyst; every passage the analyst retrieves is added to ``seen``."""
        planner_middleware: list[AgentMiddleware[Any, Any, Any]] = [
            TodoListMiddleware(),
            _read_file_only(),
            ModelCallLimitMiddleware(run_limit=PLANNER_MODEL_CALLS, exit_behavior="error"),
            *self._models.agent_middleware(),
        ]
        analyst_middleware: list[AgentMiddleware[Any, Any, Any]] = [
            _read_file_only(),
            ModelCallLimitMiddleware(run_limit=ANALYST_MODEL_CALLS, exit_behavior="error"),
            *self._models.agent_middleware(),
        ]
        return create_deep_agent(
            model=self._models.answerer,
            system_prompt=prompts.PLANNER,
            middleware=planner_middleware,
            subagents=[
                SubAgent(
                    name=ANALYST_NAME,
                    description="Filing analyst: answers ONE focused sub-question by searching "
                    "the filings; returns findings with citation keys.",
                    system_prompt=prompts.ANALYST,
                    tools=[self.search_tool(seen)],
                    model=self._models.analyst,
                    middleware=analyst_middleware,
                )
            ],
        )

    def search_tool(self, seen: dict[str, Chunk]) -> BaseTool:
        """The analyst's ``search_filings`` tool; every passage it returns is added to ``seen``."""
        index = self._index

        @tool(args_schema=_SearchArgs)
        def search_filings(query: str, document: str | None = None) -> str:
            """Search the filings and return the most relevant passages with citation keys."""
            doc_id = None if document is None else BY_SHORT_NAME[document].doc_id
            hits = [hit.chunk for hit in index.search(query, SEARCH_K, doc_id=doc_id)]
            for chunk in hits:
                seen[citation_key(chunk)] = chunk
            return format_context(hits)

        return search_filings


def _read_file_only() -> FilesystemMiddleware:
    # Replaces deepagents' default filesystem middleware (matched by name) with the minimum it
    # accepts, keeping seven unused file-tool schemas out of every prompt.
    return FilesystemMiddleware(tools=["read_file"])
