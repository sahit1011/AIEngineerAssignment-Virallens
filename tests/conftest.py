"""Shared fixtures: the real corpus and a real hybrid index, built once per session.

Paths resolve against the working directory, so run pytest from the repository root.
"""

from importlib.util import find_spec

import pytest

from virallens.common.corpus import load_corpus
from virallens.common.settings import Paths
from virallens.rag.retrieval import HybridIndex, corpus_fingerprint

# The Q2 study needs the `research` extra (torch, sentence-transformers, scikit-learn).
RESEARCH_EXTRA = find_spec("sentence_transformers") is not None and find_spec("sklearn") is not None
collect_ignore = [] if RESEARCH_EXTRA else ["test_embedder.py"]


def pytest_report_header() -> str:
    if RESEARCH_EXTRA:
        return "Q2 embedder tests: included"
    return "Q2 embedder tests: skipped (install with `uv sync --extra research`)"


@pytest.fixture(scope="session")
def paths():
    return Paths(_env_file=None)


@pytest.fixture(scope="session")
def corpus(paths):
    return load_corpus(paths.data_dir)


@pytest.fixture(scope="session")
def index(paths, corpus):
    # Real embeddings; the model is read from the cache that `main.py ingest` fills.
    return HybridIndex.build(corpus, paths.fastembed_cache_dir, corpus_fingerprint(paths.data_dir))
