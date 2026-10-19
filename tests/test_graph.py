"""Graph orchestration with scripted agents standing in for the LLM calls; retrieval is real."""

from virallens.common.chunking import Chunk
from virallens.rag.agents import Answer, Grade, Rewrite, Route, Verdict
from virallens.rag.citations import citation_key, cited_keys
from virallens.rag.graph import (
    MAX_REVISION_PASSAGES,
    NO_EVIDENCE_ANSWER,
    UNNAMED_CLAIM,
    RAGPipeline,
)
from virallens.rag.research import ResearchResult

QUESTION = "Are pure omissions actionable under Rule 10b-5(b)?"


class ScriptedAgents:
    def __init__(self, route="direct", relevant=True, answers=None, verdicts=None):
        self._route = route
        self._relevant = relevant
        self._answers = answers
        self._verdicts = list(verdicts or [Verdict(supported=True, unsupported_claims=[])])
        self.calls: list[tuple] = []

    def route(self, question):
        self.calls.append(("route",))
        return Route(route=self._route, reason="scripted")

    def grade(self, question, chunks):
        self.calls.append(("grade",))
        keys = [citation_key(c) for c in chunks[:2]] if self._relevant else []
        return Grade(relevant_keys=keys)

    def rewrite(self, question, failed_query):
        self.calls.append(("rewrite", failed_query))
        return Rewrite(query="Macquarie pure omission Rule 10b-5(b) half-truth")

    def answer(self, question, chunks):
        self.calls.append(("answer",))
        if self._answers:
            return Answer(answer=self._answers.pop(0))
        return Answer(answer=f"No, they are not [{citation_key(chunks[0])}].")

    def revise(self, question, chunks, draft, claims):
        self.calls.append(("revise", tuple(claims), draft))
        self.revision_evidence = [citation_key(c) for c in chunks]
        return Answer(answer=f"Revised: no, they are not [{citation_key(chunks[0])}].")

    def verify(self, question, answer, chunks):
        self.calls.append(("verify", tuple(citation_key(c) for c in chunks)))
        return self._verdicts.pop(0)


class ScriptedResearcher:
    def __init__(self, passages: list[Chunk], answer: str | None = None):
        self._passages = passages
        self._answer = answer
        self.questions: list[str] = []

    def run(self, question):
        self.questions.append(question)
        answer = self._answer or f"Compared [{citation_key(self._passages[0])}]."
        return ResearchResult(answer=answer, passages=self._passages)


def _steps(result) -> list[str]:
    return [step.split(":", 1)[0] for step in result.trace]


def test_direct_question_flows_through_retrieval_answer_and_verification(index):
    agents = ScriptedAgents()
    result = RAGPipeline(agents, ScriptedResearcher([]), index).ask(QUESTION)
    assert _steps(result) == ["router", "retriever", "grader", "answerer", "verifier"]
    assert result.status == "verified"
    assert result.route == "direct"
    assert [citation_key(c) for c in result.sources] == cited_keys(result.answer)


def test_no_relevant_passages_rewrites_once_then_reports_no_evidence(index):
    agents = ScriptedAgents(relevant=False)
    result = RAGPipeline(agents, ScriptedResearcher([]), index).ask(QUESTION)
    assert _steps(result) == [
        "router",
        "retriever",
        "grader",
        "rewriter",
        "retriever",
        "grader",
        "answerer",
    ]
    assert ("rewrite", QUESTION) in agents.calls
    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.status == "no_evidence"
    assert not [c for c in agents.calls if c[0] == "verify"]


def test_unsupported_answer_is_revised_once_then_returned_flagged(index):
    claim = "Pure omissions are always actionable."
    unsupported = Verdict(supported=False, unsupported_claims=[claim])
    agents = ScriptedAgents(verdicts=[unsupported, unsupported])
    result = RAGPipeline(agents, ScriptedResearcher([]), index).ask(QUESTION)
    assert _steps(result)[-4:] == ["answerer", "verifier", "answerer", "verifier"]
    revisions = [c for c in agents.calls if c[0] == "revise"]
    assert len(revisions) == 1
    assert revisions[0][1] == (claim,)
    assert revisions[0][2].startswith("No, they are not [")
    assert result.status == "unverified"
    assert result.unsupported_claims == [claim]


def test_citation_to_an_unretrieved_passage_is_caught_without_the_verifier_model(index):
    agents = ScriptedAgents(answers=["No [macquarie:9999]."])
    result = RAGPipeline(agents, ScriptedResearcher([]), index).ask(QUESTION)
    verify_calls = [c for c in agents.calls if c[0] == "verify"]
    assert len(verify_calls) == 1
    revisions = [c for c in agents.calls if c[0] == "revise"]
    assert revisions == [
        (
            "revise",
            ("Cites [macquarie:9999], which is not among the retrieved passages.",),
            "No [macquarie:9999].",
        )
    ]
    assert result.status == "verified"


def test_research_question_is_answered_by_the_deep_agent_and_verified(index, corpus):
    passages = [c for c in corpus if c.doc_id == "22-1165_10n2"][:3]
    researcher = ScriptedResearcher(passages)
    agents = ScriptedAgents(route="research")
    result = RAGPipeline(agents, researcher, index).ask(QUESTION)
    assert _steps(result) == ["router", "research", "verifier"]
    assert researcher.questions == [QUESTION]
    assert result.verified
    assert result.sources == [passages[0]]


def test_unsupported_verdict_without_named_claims_is_not_verified(index):
    unnamed = Verdict(supported=False, unsupported_claims=[])
    agents = ScriptedAgents(verdicts=[unnamed, unnamed])
    result = RAGPipeline(agents, ScriptedResearcher([]), index).ask(QUESTION)
    assert result.status == "unverified"
    assert result.unsupported_claims == [UNNAMED_CLAIM]
    assert [c[1] for c in agents.calls if c[0] == "revise"] == [(UNNAMED_CLAIM,)]


def test_verifier_model_sees_only_the_cited_passages(index, corpus):
    passages = [c for c in corpus if c.doc_id == "22-1165_10n2"][:5]
    researcher = ScriptedResearcher(passages, answer=f"Held [{citation_key(passages[2])}].")
    agents = ScriptedAgents(route="research")
    RAGPipeline(agents, researcher, index).ask(QUESTION)
    assert ("verify", (citation_key(passages[2]),)) in agents.calls


def test_research_without_passages_reports_no_evidence_and_no_stale_claims(index):
    researcher = ScriptedResearcher([], answer="Pure omissions are not actionable [macquarie:4].")
    agents = ScriptedAgents(route="research")
    result = RAGPipeline(agents, researcher, index).ask(QUESTION)
    assert _steps(result) == ["router", "research", "verifier", "answerer"]
    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.unsupported_claims == []
    assert result.status == "no_evidence"


def test_revision_sees_the_cited_passages_first_and_a_bounded_number(index, corpus):
    passages = [c for c in corpus if c.doc_id == "comp26286"][:30]
    cited = passages[25]
    researcher = ScriptedResearcher(
        passages, answer=f"The Fund breached it [{citation_key(cited)}]."
    )
    unsupported = Verdict(supported=False, unsupported_claims=["The Fund breached it."])
    agents = ScriptedAgents(route="research", verdicts=[unsupported, unsupported])
    RAGPipeline(agents, researcher, index).ask(QUESTION)
    assert agents.revision_evidence[0] == citation_key(cited)
    assert len(agents.revision_evidence) == MAX_REVISION_PASSAGES
