import json

import pytest
from pydantic import ValidationError

from virallens.common.chunking import Chunk
from virallens.common.eval_set import EvalQuestion, ExpectedSource, load_questions

CHUNK = Chunk(
    chunk_id="22-1165_10n2::0029",
    doc_id="22-1165_10n2",
    text="Pure omissions are not actionable under Rule 10b-5(b).",
    section="II",
    page_start=11,
    page_end=11,
)


def test_question_file_naming_an_unknown_filing_is_rejected(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "question": "?",
                    "route": "direct",
                    "expected": [{"doc": "nope", "pages": [1]}],
                }
            ]
        )
    )
    with pytest.raises(ValidationError, match="unknown filing 'nope'"):
        load_questions(path)


def test_hit_needs_the_expected_filing_and_an_overlapping_page():
    question = EvalQuestion(
        id="x",
        question="?",
        route="direct",
        expected=[ExpectedSource(doc="macquarie", pages=[10, 11])],
    )
    assert question.is_hit(CHUNK)
    assert not question.is_hit(Chunk(**{**CHUNK.__dict__, "page_start": 3, "page_end": 4}))
    assert not question.is_hit(Chunk(**{**CHUNK.__dict__, "doc_id": "comp26286"}))


def test_the_shipped_question_file_is_valid(paths):
    assert len(load_questions(paths.eval_questions)) == 15
