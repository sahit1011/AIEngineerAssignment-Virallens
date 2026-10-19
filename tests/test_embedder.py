"""Q2 invariants: a leak-free split, label-free training pairs, and metrics that behave."""

import random

import numpy as np
import pytest

from virallens.common.catalog import CATALOG
from virallens.embedder.cluster import assign_clusters, mean_reciprocal_rank, score
from virallens.embedder.data import (
    SHARED_SPAN_CHARS,
    ict_pairs,
    labels,
    section_neighbour_pairs,
    split_by_page_blocks,
)
from virallens.embedder.embed import lsa_encoder
from virallens.embedder.train import warmup_linear


@pytest.fixture(scope="module")
def split(corpus):
    return split_by_page_blocks(corpus)


def test_no_held_out_span_appears_in_any_training_chunk(split):
    # Brute force, independent of the split's own window sets: every span at every offset. The
    # separator cannot occur in text, so no span is matched across two training chunks.
    train_text = "\x00".join(c.text for c in split.train)
    for chunk in split.test:
        for i in range(len(chunk.text) - SHARED_SPAN_CHARS + 1):
            assert chunk.text[i : i + SHARED_SPAN_CHARS] not in train_text, chunk.chunk_id


def test_every_filing_and_topic_is_held_out(split):
    assert {c.doc_id for c in split.test} == CATALOG.keys()
    assert set(labels(split.test, "topic")) == {str(info.topic) for info in CATALOG.values()}


def test_unknown_label_kind_is_rejected(split):
    with pytest.raises(ValueError, match="unknown label kind 'case'"):
        labels(split.test, "case")


def test_ict_query_is_removed_from_its_passage(split):
    for sentence, passage in ict_pairs(split.train, random.Random(0)):
        assert sentence not in passage


def test_neighbour_pairs_are_two_apart_in_one_section(split):
    by_text = {c.text: c for c in split.train}
    pairs = section_neighbour_pairs(split.train)
    assert pairs
    for a, b in pairs:
        first, second = by_text[a], by_text[b]
        assert (first.doc_id, first.section) == (second.doc_id, second.section)
        assert second.index - first.index == 2


def test_separated_clusters_score_high_and_random_points_score_near_zero():
    rng = np.random.default_rng(0)
    centres = np.eye(3, 16)
    tight = np.vstack([centre + 0.05 * rng.standard_normal((20, 16)) for centre in centres])
    tight /= np.linalg.norm(tight, axis=1, keepdims=True)
    y = [label for label in "abc" for _ in range(20)]
    separated = score(tight, y, assign_clusters(tight, 3, seed=0))
    assert separated.silhouette > 0.8
    assert separated.ari == pytest.approx(1.0)
    assert separated.knn_purity == pytest.approx(1.0)
    noise = rng.standard_normal((60, 16))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    shuffled = score(noise, y, assign_clusters(noise, 3, seed=0))
    assert abs(shuffled.silhouette) < 0.1
    assert shuffled.similarity_gap < 0.1


def test_knn_purity_excludes_the_point_itself():
    # Three identical points per label, orthogonal labels: each point's 5 nearest neighbours are its
    # 2 twins and 3 points of the other label, so purity is exactly 2/5 (3/5 if self counted).
    vectors = np.repeat(np.eye(2), 3, axis=0)
    clusters = assign_clusters(vectors, 2, seed=0)
    assert score(vectors, list("aaabbb"), clusters).knn_purity == pytest.approx(0.4)


def test_similarity_gap_compares_distinct_pairs_only():
    # Within a label cosine 0.5, across labels 0: gap 0.5 (0.75 if self-similarity were counted).
    half = np.sqrt(3) / 2
    vectors = np.array([[1, 0, 0, 0], [0.5, half, 0, 0], [0, 0, 1, 0], [0, 0, 0.5, half]])
    clusters = assign_clusters(vectors, 2, seed=0)
    assert score(vectors, list("aabb"), clusters).similarity_gap == pytest.approx(0.5)


def test_learning_rate_warms_up_then_decays_to_zero():
    multiplier = warmup_linear(warmup=10, total=110)
    assert [multiplier(step) for step in (0, 5, 10, 60, 110)] == pytest.approx(
        [0.0, 0.5, 1.0, 0.5, 0.0]
    )


def test_mrr_uses_the_rank_of_the_first_relevant_passage():
    passages = np.eye(4)
    queries = np.array([[1.0, 0.5, 0, 0], [0, 1.0, 0.9, 0]])
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    # Query 0 ranks passage 0 first (relevant); query 1 ranks passage 1 then 2 (relevant at rank 2).
    assert mean_reciprocal_rank(queries, passages, [{0}, {2}]) == pytest.approx((1 + 0.5) / 2)


def test_lsa_refuses_text_outside_its_vocabulary(split):
    encode = lsa_encoder(split.train, seed=0)
    assert encode([split.test[0].text]).shape == (1, 128)
    with pytest.raises(ValueError, match="no vocabulary in common"):
        encode(["zzzqx vvvbn"])
