"""Q2 clustering (KMeans) and the metrics that score it, on embeddings of held-out chunks."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Neighbours per chunk for label purity: fewer than the smallest held-out topic class, so a
# perfectly separated space can score 1.0.
NEIGHBOURS = 5
# Keep the lowest-inertia of 10 KMeans runs (scikit-learn's default before 1.4 made it one run).
KMEANS_RESTARTS = 10


@dataclass(frozen=True)
class ClusteringScores:
    silhouette: float
    ari: float
    nmi: float
    knn_purity: float
    similarity_gap: float


def assign_clusters(embeddings: np.ndarray, k: int, seed: int) -> np.ndarray:
    """KMeans cluster index of each embedding."""
    # scikit-learn annotates n_init as str ("auto"), but it accepts an int.
    kmeans = KMeans(n_clusters=k, n_init=KMEANS_RESTARTS, random_state=seed)  # pyright: ignore[reportArgumentType]
    return kmeans.fit_predict(embeddings)


def score(embeddings: np.ndarray, labels: list[str], clusters: np.ndarray) -> ClusteringScores:
    """How well the embedding space separates ``labels``.

    - silhouette: of the reference labelling, cosine distance (-1..1);
    - ari / nmi: agreement of the KMeans ``clusters`` with the reference labelling;
    - knn_purity: share of each chunk's NEIGHBOURS nearest neighbours with its label;
    - similarity_gap: mean cosine within a label minus mean cosine across labels.
    """
    y = np.array(labels)
    similarity = embeddings @ embeddings.T
    same = y[:, None] == y[None, :]
    off_diagonal = ~np.eye(len(y), dtype=bool)
    np.fill_diagonal(similarity, -np.inf)
    neighbours = np.argsort(-similarity, axis=1)[:, :NEIGHBOURS]
    np.fill_diagonal(similarity, 1.0)
    return ClusteringScores(
        silhouette=float(silhouette_score(embeddings, y, metric="cosine")),
        ari=float(adjusted_rand_score(y, clusters)),
        nmi=float(normalized_mutual_info_score(y, clusters)),
        knn_purity=float((y[neighbours] == y[:, None]).mean()),
        similarity_gap=float(similarity[same & off_diagonal].mean() - similarity[~same].mean()),
    )


def cluster_composition(labels: list[str], clusters: np.ndarray) -> list[dict[str, int]]:
    """Label counts inside each cluster, largest cluster first."""
    composition = [
        dict(Counter(label for label, c in zip(labels, clusters, strict=True) if c == k))
        for k in sorted(set(clusters.tolist()))
    ]
    return sorted(composition, key=lambda counts: -sum(counts.values()))


def mean_reciprocal_rank(
    query_vectors: np.ndarray, passage_vectors: np.ndarray, relevant: list[set[int]]
) -> float:
    """MRR of the first relevant passage for each query, ranking passages by cosine similarity."""
    ranks = np.argsort(-(query_vectors @ passage_vectors.T), axis=1)
    reciprocal = []
    for order, targets in zip(ranks, relevant, strict=True):
        positions = [rank for rank, i in enumerate(order, start=1) if i in targets]
        reciprocal.append(1 / positions[0] if positions else 0.0)
    return float(np.mean(reciprocal))
