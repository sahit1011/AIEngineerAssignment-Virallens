import shutil
from pathlib import Path

import pytest
from pypdf import PdfReader

from virallens.common.catalog import CATALOG
from virallens.common.chunking import CHUNK_SIZE
from virallens.common.corpus import load_corpus

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def chunks():
    return load_corpus(DATA_DIR)


def test_every_catalogued_document_is_chunked(chunks):
    assert {c.doc_id for c in chunks} == CATALOG.keys()
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_page_furniture_and_index_pages_are_gone(chunks):
    text = "\n".join(c.text for c in chunks)
    assert "PageID" not in text
    assert "....." not in text
    assert "Opinion of the Court" not in text


def test_chunks_are_bounded_and_page_ranges_are_valid(chunks):
    page_counts = {doc_id: len(PdfReader(DATA_DIR / f"{doc_id}.pdf").pages) for doc_id in CATALOG}
    for c in chunks:
        assert 0 < len(c.text) <= CHUNK_SIZE
        assert 1 <= c.page_start <= c.page_end <= page_counts[c.doc_id]


def test_known_passage_is_cited_to_its_page(chunks):
    # Macquarie slip opinion, page 1 of the syllabus.
    hits = [c for c in chunks if "Macquarie’s stock price fell 41%" in c.text]
    assert [(c.doc_id, c.page_start) for c in hits] == [("22-1165_10n2", 1)]


def test_directory_not_matching_catalog_is_rejected(tmp_path):
    shutil.copy(DATA_DIR / "comp26286.pdf", tmp_path / "unexpected.pdf")
    with pytest.raises(ValueError, match="unlisted=\\['unexpected'\\]"):
        load_corpus(tmp_path)
