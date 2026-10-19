"""Q2 embedding: the three encoders compared, each mapping texts to L2-normalised vectors.

- ``baseline``: ``all-MiniLM-L6-v2`` as published (384-d), the general-purpose reference;
- ``lsa``: TF-IDF + truncated SVD fit on the training split (128-d), the classical domain model;
- ``domain``: the fine-tuned MiniLM with a 128-d head (``train.py``).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from virallens.common.chunking import Chunk
from virallens.embedder.train import DOMAIN_DIM, load_base

Encoder = Callable[[list[str]], np.ndarray]

# Same width as the domain model, so the comparison is not a question of dimensionality.
LSA_DIM = DOMAIN_DIM
# Terms in a single training chunk carry no co-occurrence signal for SVD; sublinear tf damps
# terms repeated within one chunk (Manning, Raghavan & Schütze, Introduction to IR, §6.4.1).
LSA_MIN_DF = 2


def sentence_encoder(model: SentenceTransformer) -> Encoder:
    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    return encode


def baseline_encoder(cache_dir: Path) -> Encoder:
    return sentence_encoder(load_base(cache_dir))


def lsa_encoder(train: list[Chunk], seed: int) -> Encoder:
    lsa = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=LSA_MIN_DF, stop_words="english"),
        TruncatedSVD(n_components=LSA_DIM, random_state=seed),
    )
    lsa.fit([chunk.text for chunk in train])

    def encode(texts: list[str]) -> np.ndarray:
        vectors = lsa.transform(texts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        empty = [text[:60] for text, norm in zip(texts, norms[:, 0], strict=True) if norm == 0]
        if empty:
            raise ValueError(f"LSA has no vocabulary in common with: {empty}")
        return vectors / norms

    return encode
