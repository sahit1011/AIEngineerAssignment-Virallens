"""The hand-written evaluation questions and the rule for judging a chunk against them.

Each question names the filing(s) and pages where its answer was located by reading the PDFs. A
chunk is a hit when it comes from an expected filing and its page range overlaps an expected page.
Shared by the RAG evaluation (Q1) and the embedder study's retrieval check (Q2).
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, TypeAdapter, field_validator

from virallens.common.catalog import BY_SHORT_NAME
from virallens.common.chunking import Chunk


class ExpectedSource(BaseModel):
    doc: str
    pages: list[int]

    @field_validator("doc")
    @classmethod
    def _known_filing(cls, doc: str) -> str:
        if doc not in BY_SHORT_NAME:
            raise ValueError(f"unknown filing {doc!r}; expected one of {sorted(BY_SHORT_NAME)}")
        return doc


class EvalQuestion(BaseModel):
    id: str
    question: str
    route: Literal["direct", "research"]
    expected: list[ExpectedSource]

    def is_hit(self, chunk: Chunk) -> bool:
        pages = set(range(chunk.page_start, chunk.page_end + 1))
        return any(
            BY_SHORT_NAME[source.doc].doc_id == chunk.doc_id and pages & set(source.pages)
            for source in self.expected
        )


def load_questions(path: Path) -> list[EvalQuestion]:
    return TypeAdapter(list[EvalQuestion]).validate_python(json.loads(path.read_text()))
