"""Hybrid retrieval over the chunked filings: BM25 + dense vectors, fused by Reciprocal Rank Fusion.

Legal questions hinge on exact strings ("Item 105", "Rule 10b-5(b)", "§20(a)") that dense
embeddings blur, and on paraphrase ("failing to mention past breaches") that BM25 misses; fusing
both ranks covers each one's blind spot without tuning score scales against each other.
"""

import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from virallens.common import catalog, chunking, cleaning
from virallens.common.catalog import CATALOG
from virallens.common.chunking import Chunk

# 384-d, 512-token input, 67 MB ONNX model; chunks of at most 1000 characters fit its input.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# The constant of Cormack, Clarke & Büttcher (SIGIR 2009), which introduced Reciprocal Rank Fusion.
RRF_K = 60
# Each retriever proposes this many candidates before fusion; 20 of 370 chunks is enough depth for
# a chunk ranked mid-list by one retriever to surface through the other.
CANDIDATES_PER_RETRIEVER = 20
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.§()][a-z0-9]+)*")
_CHUNKS_FILE = "chunks.json"
_EMBEDDINGS_FILE = "embeddings.npy"
_MANIFEST_FILE = "manifest.json"


Retriever = Literal["bm25", "dense"]
HYBRID: tuple[Retriever, ...] = ("bm25", "dense")


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


def indexed_text(chunk: Chunk) -> str:
    """The text both retrievers see: the chunk prefixed with its filing and section."""
    return f"{CATALOG[chunk.doc_id].title}\nSection: {chunk.section}\n\n{chunk.text}"


def corpus_fingerprint(data_dir: Path) -> str:
    """What an index is built from: the PDFs, the code that cleans and chunks them, the text indexed
    per chunk and the embedding model. Changing any of them makes an existing index stale."""
    digest = hashlib.sha256(EMBEDDING_MODEL.encode())
    for source in (cleaning, chunking, catalog, indexed_text):
        digest.update(inspect.getsource(source).encode())
    for pdf in sorted(data_dir.glob("*.pdf")):
        digest.update(pdf.name.encode())
        digest.update(pdf.read_bytes())
    return digest.hexdigest()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HybridIndex:
    def __init__(
        self, chunks: list[Chunk], embeddings: np.ndarray, embedder: TextEmbedding, fingerprint: str
    ):
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        self._chunks = chunks
        self._embeddings = embeddings
        self._embedder = embedder
        self._bm25 = BM25Okapi([_tokens(indexed_text(chunk)) for chunk in chunks])
        self.fingerprint = fingerprint

    @classmethod
    def build(cls, chunks: list[Chunk], cache_dir: Path, fingerprint: str) -> "HybridIndex":
        embedder = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(cache_dir))
        vectors = np.array(list(embedder.passage_embed([indexed_text(c) for c in chunks])))
        return cls(chunks, _normalise(vectors), embedder, fingerprint)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / _CHUNKS_FILE).write_text(json.dumps([asdict(c) for c in self._chunks]))
        np.save(index_dir / _EMBEDDINGS_FILE, self._embeddings)
        (index_dir / _MANIFEST_FILE).write_text(json.dumps({"fingerprint": self.fingerprint}))

    @classmethod
    def load(cls, index_dir: Path, cache_dir: Path, expected_fingerprint: str) -> "HybridIndex":
        manifest_path = index_dir / _MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"{index_dir}: no index found; run `python main.py ingest`")
        # A manifest without a fingerprint was not written by `ingest`: treat it as stale too.
        fingerprint = json.loads(manifest_path.read_text()).get("fingerprint")
        if fingerprint != expected_fingerprint:
            raise ValueError(
                f"{index_dir} was built from other filings, chunking or embedding model; "
                "run `python main.py ingest`"
            )
        chunks = [Chunk(**c) for c in json.loads((index_dir / _CHUNKS_FILE).read_text())]
        embeddings = np.load(index_dir / _EMBEDDINGS_FILE)
        embedder = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(cache_dir))
        return cls(chunks, embeddings, embedder, fingerprint)

    def search(
        self,
        query: str,
        k: int,
        doc_id: str | None = None,
        retrievers: tuple[Retriever, ...] = HYBRID,
    ) -> list[SearchHit]:
        """Top-k chunks for ``query``; ``retrievers`` narrows fusion to one side for evaluation."""
        if doc_id is not None and doc_id not in CATALOG:
            raise ValueError(f"unknown doc_id {doc_id!r}; expected one of {sorted(CATALOG)}")
        allowed = np.array([doc_id is None or c.doc_id == doc_id for c in self._chunks])
        fused: dict[int, float] = {}
        for retriever in retrievers:
            scores = self._scores(retriever, query)
            # A BM25 score of 0 means no query term occurs in the chunk: not a candidate, however
            # the tie among zeros happens to sort.
            eligible = allowed & (scores > 0) if retriever == "bm25" else allowed
            ranked = [i for i in np.argsort(-scores) if eligible[i]][:CANDIDATES_PER_RETRIEVER]
            for rank, i in enumerate(ranked, start=1):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank)
        best = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        return [SearchHit(self._chunks[i], fused[i]) for i in best]

    def _scores(self, retriever: Retriever, query: str) -> np.ndarray:
        if retriever == "bm25":
            return self._bm25.get_scores(_tokens(query))
        query_vector = _normalise(np.array(list(self._embedder.query_embed(query))))[0]
        return self._embeddings @ query_vector


def _normalise(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
