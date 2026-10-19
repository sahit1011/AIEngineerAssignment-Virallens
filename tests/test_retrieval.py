import json
import shutil

import pytest

from virallens.rag.citations import citation_key
from virallens.rag.graph import RETRIEVE_K
from virallens.rag.retrieval import HybridIndex, corpus_fingerprint


def test_terse_rule_number_query_reaches_the_answerer(index):
    # BM25 ranks Last Atlantis first; dense reads "Rule 56(f)" as near "Rule 10b-5" and ranks the
    # briefs first, so after fusion Last Atlantis sits at rank 5: inside what the answerer sees.
    hits = index.search("Rule 56(f) affidavit additional discovery", RETRIEVE_K)
    assert "USCOURTS-ilnd-1_04-cv-00397-5" in {hit.chunk.doc_id for hit in hits}
    bm25 = index.search("Rule 56(f) affidavit additional discovery", 1, retrievers=("bm25",))
    assert bm25[0].chunk.doc_id == "USCOURTS-ilnd-1_04-cv-00397-5"


def test_bm25_proposes_only_chunks_containing_a_query_term(index):
    # Two chunks contain "disgorge"; the other 368 score 0 and must not fill the candidate list.
    hits = index.search("disgorge", 20, retrievers=("bm25",))
    assert len(hits) == 2
    assert all("disgorge" in hit.chunk.text.lower() for hit in hits)


def test_document_filter_restricts_every_hit(index):
    hits = index.search("risk factors", 5, doc_id="comp26286")
    assert len(hits) == 5
    assert {hit.chunk.doc_id for hit in hits} == {"comp26286"}


def test_unknown_document_is_rejected(index):
    with pytest.raises(ValueError, match="unknown doc_id 'nope'"):
        index.search("risk", 3, doc_id="nope")


def test_hybrid_scores_fuse_both_retrievers(index):
    query = "pure omissions Rule 10b-5(b)"
    hybrid = index.search(query, 1)[0]
    single = max(
        index.search(query, 1, retrievers=("bm25",))[0].score,
        index.search(query, 1, retrievers=("dense",))[0].score,
    )
    assert hybrid.score > single


def test_saved_index_loads_with_identical_results(index, tmp_path, paths):
    index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path, paths.fastembed_cache_dir, index.fingerprint)
    query = "concentration policy 25% of total assets"
    assert [citation_key(h.chunk) for h in loaded.search(query, 5)] == [
        citation_key(h.chunk) for h in index.search(query, 5)
    ]


def test_stale_index_is_rejected(index, tmp_path, paths):
    index.save(tmp_path)
    with pytest.raises(ValueError, match="run `python main.py ingest`"):
        HybridIndex.load(tmp_path, paths.fastembed_cache_dir, "fingerprint-of-other-filings")


def test_manifest_without_a_fingerprint_is_rejected_as_stale(index, tmp_path, paths):
    index.save(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"embedding_model": "BAAI/bge-small-en-v1.5"})
    )
    with pytest.raises(ValueError, match="run `python main.py ingest`"):
        HybridIndex.load(tmp_path, paths.fastembed_cache_dir, index.fingerprint)


def test_fingerprint_changes_when_a_filing_changes(tmp_path, paths):
    shutil.copytree(paths.data_dir, tmp_path / "data")
    with (tmp_path / "data" / "comp26286.pdf").open("ab") as pdf:
        pdf.write(b"\n% amended")
    assert corpus_fingerprint(tmp_path / "data") != corpus_fingerprint(paths.data_dir)


def test_missing_index_names_the_ingest_command(tmp_path, paths):
    with pytest.raises(FileNotFoundError, match="main.py ingest"):
        HybridIndex.load(tmp_path, paths.fastembed_cache_dir, "any")
