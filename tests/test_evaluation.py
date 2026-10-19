import groq
import httpx
import pytest

from virallens.common.chunking import Chunk
from virallens.common.eval_set import EvalQuestion, ExpectedSource
from virallens.rag.evaluation import (
    EndToEndReport,
    EndToEndRow,
    evaluate_end_to_end,
    evaluate_retrieval,
)
from virallens.rag.graph import RAGResult

CHUNK = Chunk(
    chunk_id="22-1165_10n2::0029",
    doc_id="22-1165_10n2",
    text="Pure omissions are not actionable under Rule 10b-5(b).",
    section="II",
    page_start=11,
    page_end=11,
)


def _question(id_, text, doc, pages):
    return EvalQuestion(
        id=id_, question=text, route="direct", expected=[ExpectedSource(doc=doc, pages=pages)]
    )


def test_retrieval_metrics_count_the_rank_of_the_first_hit(index):
    query = "Rule 56(f) affidavit additional discovery denied"
    questions = [
        _question("found", query, "last-atlantis", [1, 19, 20]),
        _question("absent", query, "sec-v-chiueh", [50]),
    ]
    report = evaluate_retrieval(index, questions, 3, ("bm25", "dense"))
    assert report.hit_at_k == 0.5
    assert report.misses == ["absent"]
    assert 0 < report.mrr <= 0.5


class _PipelineFailingOn:
    """Stands in for RAGPipeline: evaluate_end_to_end only calls ask()."""

    def __init__(self, failing_question, error, result):
        self._failing, self._error, self._result = failing_question, error, result

    def ask(self, question):
        if question == self._failing:
            raise self._error
        return self._result


def test_a_failed_question_is_recorded_and_the_rest_still_run():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = groq.BadRequestError(
        "scripted", response=httpx.Response(400, request=request), body=None
    )
    result = RAGResult("q", "direct", "No [macquarie:29].", "verified", [], [CHUNK], [], {})
    questions = [
        _question("a", "first", "macquarie", [11]),
        _question("b", "second", "macquarie", [11]),
    ]
    report = evaluate_end_to_end(_PipelineFailingOn("first", error, result), questions)
    assert [f.id for f in report.failures] == ["a"]
    assert report.failures[0].error.startswith("BadRequestError")
    assert [row.id for row in report.rows] == ["b"]
    assert report.rows[0].cites_expected_source
    assert report.rows[0].cited == ["macquarie:29 p. 11"]


def _row(route, status, cites, tokens):
    return EndToEndRow("q", "direct", route, status, cites, [], tokens)


def test_report_rates_are_over_answered_questions():
    report = EndToEndReport(
        rows=[
            _row("direct", "verified", True, 100),
            _row("research", "unverified", True, 300),
            _row("direct", "no_evidence", False, 50),
            _row("direct", "verified", True, 150),
        ],
        failures=[],
    )
    assert report.route_accuracy == 0.75
    assert report.verified_rate == 0.5
    assert report.cites_expected_rate == 0.75
    assert report.total_tokens == 600


def test_rates_of_a_report_with_no_answers_are_an_error():
    with pytest.raises(ValueError, match="no answered questions"):
        _ = EndToEndReport(rows=[], failures=[]).verified_rate
