"""The DeepAgents research branch is built as configured: one delegate, bounded, with fallback."""

import pytest
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents.middleware import ModelCallLimitMiddleware
from pydantic import ValidationError

from virallens.common.chunking import Chunk
from virallens.rag.citations import cited_keys
from virallens.rag.config import Settings
from virallens.rag.llm import GroqCallPolicy, Models
from virallens.rag.research import ANALYST_NAME, Researcher


@pytest.fixture
def built_subagents(monkeypatch, index):
    """The sub-agent specs create_deep_agent finally hands to its SubAgentMiddleware.

    deepagents merges our specs with its defaults internally and exposes no view of the result, so
    the specs are captured where it constructs the middleware (deepagents is pinned in uv.lock).
    """

    def build(openrouter_key):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        settings = Settings(
            _env_file=None, groq_api_key="gsk-test", openrouter_api_key=openrouter_key
        )
        captured = {}
        original = SubAgentMiddleware.__init__

        def capture(self, *args, **kwargs):
            captured["subagents"] = kwargs["subagents"]
            original(self, *args, **kwargs)

        monkeypatch.setattr(SubAgentMiddleware, "__init__", capture)
        Researcher(Models(settings), index).agent({})
        monkeypatch.setattr(SubAgentMiddleware, "__init__", original)
        return captured["subagents"]

    return build


def _tools(spec) -> set[str]:
    own = {tool.name for tool in spec["tools"]}
    return own | {tool.name for m in spec["middleware"] for tool in getattr(m, "tools", [])}


def test_the_filing_analyst_is_the_planners_only_delegate(built_subagents):
    assert [spec["name"] for spec in built_subagents(None)] == [ANALYST_NAME]


def test_the_analyst_can_only_search_and_read(built_subagents):
    (analyst,) = built_subagents(None)
    assert _tools(analyst) == {"search_filings", "read_file"}


def test_the_analyst_is_bounded_and_falls_back_when_openrouter_is_configured(built_subagents):
    (analyst,) = built_subagents("sk-or-test")
    kinds = [type(m) for m in analyst["middleware"]]
    assert ModelCallLimitMiddleware in kinds
    assert GroqCallPolicy in kinds


def test_search_tool_records_every_passage_it_returns(index):
    seen: dict[str, Chunk] = {}
    search = Researcher(models=None, index=index).search_tool(seen)
    output = search.invoke({"query": "concentration policy 25%", "document": "sec-v-chiueh"})
    assert set(cited_keys(output)) == set(seen)
    assert {chunk.doc_id for chunk in seen.values()} == {"comp26286"}


def test_search_tool_rejects_an_unknown_filing_name(index):
    search = Researcher(models=None, index=index).search_tool({})
    with pytest.raises(ValidationError):
        search.invoke({"query": "risk", "document": "not-a-filing"})
