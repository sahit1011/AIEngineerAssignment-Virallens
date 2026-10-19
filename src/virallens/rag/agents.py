"""Single-purpose agents of the LangGraph flow: router, grader, rewriter, answerer, verifier.

Each is one structured LLM call. The classification-style roles (route, grade, rewrite, verify) run
on the classifier model and answering on the answer model, so they draw on separate Groq quotas.
"""

from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from virallens.common.chunking import Chunk
from virallens.rag import prompts
from virallens.rag.citations import format_context, normalise_citations
from virallens.rag.llm import Models


class Route(BaseModel):
    route: Literal["direct", "research"] = Field(
        description="direct unless the question asks to compare or connect filings or parties"
    )
    reason: str = Field(description="One sentence")


class Grade(BaseModel):
    relevant_keys: list[str] = Field(
        description="Keys of the passages that help answer the question; empty if none do"
    )


class Rewrite(BaseModel):
    query: str = Field(description="A search query likely to find the missing passages")


class Answer(BaseModel):
    answer: str = Field(description="The answer, citing passage keys inline like [macquarie:12]")


class Verdict(BaseModel):
    supported: bool = Field(description="True if every claim is backed by its cited passages")
    unsupported_claims: list[str] = Field(
        description="Each claim not backed by the passages it cites; empty if supported"
    )


class Agents:
    def __init__(self, models: Models):
        self._router = models.structured(models.classifier, Route)
        self._grader = models.structured(models.classifier, Grade)
        self._rewriter = models.structured(models.classifier, Rewrite)
        self._verifier = models.structured(models.classifier, Verdict)
        self._answerer = models.structured(models.answerer, Answer)

    def route(self, question: str) -> Route:
        return self._router.invoke(_messages(prompts.ROUTER, question))

    def grade(self, question: str, chunks: list[Chunk]) -> Grade:
        prompt = f"Question: {question}\n\nPassages:\n\n{format_context(chunks)}"
        return self._grader.invoke(_messages(prompts.GRADER, prompt))

    def rewrite(self, question: str, failed_query: str) -> Rewrite:
        prompt = f"Question: {question}\nQuery that found nothing useful: {failed_query}"
        return self._rewriter.invoke(_messages(prompts.REWRITER, prompt))

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        prompt = f"Passages:\n\n{format_context(chunks)}\n\nQuestion: {question}"
        return self._normalised(self._answerer.invoke(_messages(prompts.ANSWERER, prompt)))

    def revise(self, question: str, chunks: list[Chunk], draft: str, claims: list[str]) -> Answer:
        issues = "\n".join(f"- {claim}" for claim in claims)
        prompt = (
            f"Passages:\n\n{format_context(chunks)}\n\nQuestion: {question}\n\n"
            f"Draft answer:\n{draft}\n\nClaims the verifier found unsupported:\n{issues}"
        )
        return self._normalised(self._answerer.invoke(_messages(prompts.REVISER, prompt)))

    def verify(self, question: str, answer: str, chunks: list[Chunk]) -> Verdict:
        prompt = (
            f"Passages:\n\n{format_context(chunks)}\n\nQuestion: {question}\n\nAnswer:\n{answer}"
        )
        return self._verifier.invoke(_messages(prompts.VERIFIER, prompt))

    @staticmethod
    def _normalised(answer: Answer) -> Answer:
        return Answer(answer=normalise_citations(answer.answer))


def _messages(system: str, human: str) -> list[BaseMessage]:
    return [SystemMessage(system), HumanMessage(human)]
