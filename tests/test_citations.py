from virallens.common.chunking import Chunk
from virallens.rag.citations import (
    citation_key,
    cited_keys,
    format_context,
    normalise_citations,
)

CHUNK = Chunk(
    chunk_id="22-1165_10n2::0029",
    doc_id="22-1165_10n2",
    text="Pure omissions are not actionable under Rule 10b-5(b).",
    section="II",
    page_start=11,
    page_end=11,
)


def test_citation_key_uses_the_short_filing_name_and_chunk_index():
    assert citation_key(CHUNK) == "macquarie:29"


def test_native_brackets_become_parseable_citations():
    text = normalise_citations(
        "Not actionable 【macquarie:4】【macquarie:29】 per [fb-petitioners:6]."
    )
    assert cited_keys(text) == ["macquarie:4", "macquarie:29", "fb-petitioners:6"]


def test_native_brackets_with_a_line_suffix_are_normalised():
    assert cited_keys(normalise_citations("Held 【macquarie:4†L1-L3】.")) == ["macquarie:4"]


def test_every_key_in_a_combined_bracket_is_cited():
    text = "Both agree [fb-petitioners:6, macquarie:9999; fb-united-states:21] but [sic] not [2]."
    assert cited_keys(text) == ["fb-petitioners:6", "macquarie:9999", "fb-united-states:21"]


def test_context_block_carries_key_filing_heading_and_pages():
    assert format_context([CHUNK]).splitlines()[0] == (
        "[macquarie:29] Macquarie Infrastructure Corp. v. Moab Partners, L.P. (slip opinion) | "
        "II | p. 11"
    )
    assert Chunk(**{**CHUNK.__dict__, "page_end": 12}).pages == "pp. 11-12"
