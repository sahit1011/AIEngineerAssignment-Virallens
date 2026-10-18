import re

import pytest

from virallens.common.chunking import CHUNK_SIZE, FRONT_MATTER_SECTION, chunk_document
from virallens.common.cleaning import Line

SENTENCE = "The district court dismissed the complaint for failure to plead falsity."


def _paragraph(page: int, sentences: int) -> list[Line]:
    return [Line(page, SENTENCE) for _ in range(sentences)]


def test_all_caps_headings_split_sections_and_page_one_is_front_matter():
    lines = [
        Line(1, "FACEBOOK, INC., ET AL., PETITIONERS"),
        *_paragraph(1, 2),
        Line(2, "SUMMARY OF ARGUMENT"),
        *_paragraph(2, 2),
        Line(3, "ARGUMENT"),
        *_paragraph(3, 2),
    ]
    chunks = chunk_document("doc", lines)
    assert [(c.section, c.page_start) for c in chunks] == [
        (FRONT_MATTER_SECTION, 1),
        ("SUMMARY OF ARGUMENT", 2),
        ("ARGUMENT", 3),
    ]
    assert chunks[0].text.startswith("FACEBOOK, INC., ET AL., PETITIONERS ")


def test_consecutive_heading_lines_form_one_heading():
    lines = [
        *_paragraph(1, 2),
        Line(2, "STATUTORY AND REGULATORY PROVISIONS"),
        Line(2, "INVOLVED"),
        *_paragraph(2, 2),
    ]
    assert (
        chunk_document("doc", lines)[-1].section == "STATUTORY AND REGULATORY PROVISIONS INVOLVED"
    )


def test_roman_numeral_headings_split_sections():
    lines = [
        *_paragraph(1, 2),
        Line(2, "I. Factual Background"),
        *_paragraph(2, 2),
        Line(3, "II"),
        *_paragraph(3, 2),
    ]
    assert [c.section for c in chunk_document("doc", lines)][1:] == ["I. Factual Background", "II"]


def test_signature_fragment_is_folded_into_previous_section_with_its_text():
    lines = [
        *_paragraph(1, 2),
        Line(9, "CONCLUSION"),
        Line(9, "The judgment of the court of appeals should be affirmed. Respectfully submitted."),
        Line(9, "MEGAN BARBERO"),
        Line(9, "General Counsel"),
    ]
    last = chunk_document("doc", lines)[-1]
    assert last.section == "CONCLUSION"
    assert last.text.endswith("Respectfully submitted. MEGAN BARBERO General Counsel")


@pytest.mark.parametrize(("pages", "lines_per_page"), [(range(2, 5), 15), (range(2, 42), 1)])
def test_each_chunk_is_cited_to_the_pages_of_its_first_and_last_sentence(pages, lines_per_page):
    # Equal-width one-sentence lines join with single spaces, so chunks start and end exactly on
    # line boundaries; with one line per page every chunk boundary is also a page boundary.
    lines = [
        Line(
            page, f"Line {k:02d} of page {page} states one proposition about Rule 10b-5 liability."
        )
        for page in pages
        for k in range(lines_per_page)
    ]
    chunks = chunk_document("doc", [*_paragraph(1, 2), Line(2, "ARGUMENT"), *lines])
    argument = [c for c in chunks if c.section == "ARGUMENT"]
    assert len(argument) > 3
    for chunk in argument:
        cited = [int(p) for p in re.findall(r"of page (\d+)", chunk.text)]
        assert len(chunk.text) <= CHUNK_SIZE
        assert (chunk.page_start, chunk.page_end) == (cited[0], cited[-1])
    assert len({c.page_start for c in argument}) > 1
    assert any(c.page_start != c.page_end for c in argument)


def test_chunk_ids_are_unique_and_ordered():
    lines = [*_paragraph(1, 40)]
    ids = [c.chunk_id for c in chunk_document("doc", lines)]
    assert ids == sorted(set(ids))
    assert ids[0] == "doc::0000"


def _sections_of(lines: list[Line]) -> list[str]:
    return list(dict.fromkeys(c.section for c in chunk_document("doc", lines)))


def test_title_case_point_headings_nest_under_their_parent():
    lines = [
        *_paragraph(1, 2),
        Line(2, "ARGUMENT"),
        *_paragraph(2, 2),
        Line(2, "C. The Ninth Circuit’s Approach Is Erroneous"),
        *_paragraph(2, 2),
        Line(3, "1. The Ninth Circuit’s Approach Ignores The Actual"),
        *_paragraph(3, 2),
        Line(4, "D. Petitioners Would Prevail Even Under The Mistaken"),
        *_paragraph(4, 2),
    ]
    assert _sections_of(lines)[1:] == [
        "ARGUMENT",
        "ARGUMENT > C. The Ninth Circuit’s Approach Is Erroneous",
        "ARGUMENT > C. The Ninth Circuit’s Approach Is Erroneous > "
        "1. The Ninth Circuit’s Approach Ignores The Actual",
        "ARGUMENT > D. Petitioners Would Prevail Even Under The Mistaken",
    ]


def test_chicago_style_lettered_heading_is_detected():
    lines = [
        *_paragraph(1, 2),
        Line(2, "II. Analysis"),
        *_paragraph(2, 2),
        Line(2, "C. Claims under Rule 10b-5(a) and (c)"),
        *_paragraph(2, 2),
    ]
    assert _sections_of(lines)[1:] == [
        "II. Analysis",
        "II. Analysis > C. Claims under Rule 10b-5(a) and (c)",
    ]


def test_sentence_case_and_proper_noun_openings_stay_in_the_body():
    lines = [
        *_paragraph(1, 2),
        Line(2, "SUMMARY OF ARGUMENT"),
        *_paragraph(2, 1),
        Line(2, "D. Unlike the Ninth Circuit, some courts of appeals"),
        *_paragraph(2, 1),
        Line(2, "1. Under the Securities Act of 1933 (Securities Act) ,"),
        *_paragraph(2, 1),
    ]
    assert _sections_of(lines)[1:] == ["SUMMARY OF ARGUMENT"]


def test_lettered_line_continuing_a_sentence_is_not_a_heading():
    lines = [
        *_paragraph(1, 2),
        Line(2, "ARGUMENT"),
        Line(2, "Securities Fraud is described in 1 Alan R. Bromberg & Lewis"),
        Line(2, "D. Lowenfels, Bromberg and Lowenfels on Securities"),
        *_paragraph(2, 2),
    ]
    assert _sections_of(lines)[1:] == ["ARGUMENT"]


def test_heading_after_footnote_marker_is_detected():
    lines = [
        *_paragraph(1, 2),
        Line(2, "STATEMENT"),
        *_paragraph(2, 1),
        Line(2, "Ibid.1"),
        Line(2, "B. Facts And Procedural History"),
        *_paragraph(2, 2),
    ]
    assert _sections_of(lines)[1:] == ["STATEMENT", "STATEMENT > B. Facts And Procedural History"]


def test_wrapped_heading_continues_but_short_heading_does_not_swallow_body():
    wrapped = [
        Line(2, "A. Risk Disclosures Under Item 105 Make No Implied Representations"),
        Line(2, "About A Company’s Past Experiences"),
    ]
    short = [
        Line(3, "B. Rule 10b-5"),
        Line(3, "Section 10(b) Of The Exchange Act Makes Deception Unlawful."),
    ]
    lines = [
        *_paragraph(1, 2),
        Line(2, "ARGUMENT"),
        *_paragraph(2, 2),
        *wrapped,
        *_paragraph(2, 2),
        *short,
        *_paragraph(3, 2),
    ]
    assert _sections_of(lines)[1:] == [
        "ARGUMENT",
        "ARGUMENT > A. Risk Disclosures Under Item 105 Make No Implied Representations About A "
        "Company’s Past Experiences",
        "ARGUMENT > B. Rule 10b-5",
    ]
