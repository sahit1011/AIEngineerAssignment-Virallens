"""Q2 study: domain embedder vs. baseline vs. LSA on held-out pages of the filings.

    python -m virallens.embedder                 # 3 seeds, writes research/outputs/

Every score is computed on test-split chunks only; retrieval ranks held-out passages for the eval
questions whose answering passages are held out. The domain model and LSA are fit on the train
split; the baseline is used as published. Seeds vary the fine-tuning, LSA's randomised SVD and the
KMeans initialisation; the baseline's embeddings are fixed, so its spread comes from KMeans alone.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from virallens.common.catalog import CATALOG
from virallens.common.chunking import Chunk
from virallens.common.corpus import load_corpus
from virallens.common.eval_set import load_questions
from virallens.common.settings import Paths
from virallens.embedder.cluster import (
    ClusteringScores,
    assign_clusters,
    cluster_composition,
    mean_reciprocal_rank,
    score,
)
from virallens.embedder.data import Split, labels, split_by_page_blocks
from virallens.embedder.embed import Encoder, baseline_encoder, lsa_encoder, sentence_encoder
from virallens.embedder.plots import metric_bars, similarity_heatmaps, tsne_maps
from virallens.embedder.train import train_domain_model

logger = logging.getLogger(__name__)

SEEDS = (0, 1, 2)
LABEL_KINDS = ("topic", "doc_type", "doc")
MODELS = ("baseline", "lsa", "domain")
# Neighbours listed per probe in the qualitative check.
PROBE_NEIGHBOURS = 3


@dataclass(frozen=True)
class ScoreRow:
    model: str
    seed: int
    labels: str
    scores: ClusteringScores


@dataclass(frozen=True)
class StudyResults:
    split: Split
    rows: list[ScoreRow]
    retrieval_mrr: dict[str, list[float]]
    retrieval_questions: list[str]
    epoch_losses: dict[int, list[float]]
    # Seed-0 embeddings of the test chunks and their KMeans topic clusters, for the figures, the
    # neighbour check and the cluster-composition table.
    test_embeddings: dict[str, np.ndarray]
    topic_clusters: dict[str, np.ndarray]


def run_study(paths: Paths, seeds: tuple[int, ...]) -> StudyResults:
    chunks = load_corpus(paths.data_dir)
    split = split_by_page_blocks(chunks)
    # Retrieval is scored only where it cannot reward memorisation: questions answered by at least
    # one held-out passage, ranked against held-out passages only.
    relevant_by_question = {
        q.id: (q, {i for i, c in enumerate(split.test) if q.is_hit(c)})
        for q in load_questions(paths.eval_questions)
    }
    retrieval = [(q, hits) for q, hits in relevant_by_question.values() if hits]
    test_texts = [c.text for c in split.test]
    baseline = baseline_encoder(paths.hf_cache_dir)

    rows: list[ScoreRow] = []
    mrr: dict[str, list[float]] = {model: [] for model in MODELS}
    losses: dict[int, list[float]] = {}
    figures: dict[str, np.ndarray] = {}
    topic_clusters: dict[str, np.ndarray] = {}
    for seed in seeds:
        logger.info("seed %d: training domain model on %d chunks", seed, len(split.train))
        trained = train_domain_model(split.train, seed, paths.hf_cache_dir)
        losses[seed] = trained.epoch_losses
        encoders: dict[str, Encoder] = {
            "baseline": baseline,
            "lsa": lsa_encoder(split.train, seed),
            "domain": sentence_encoder(trained.model),
        }
        for model, encode in encoders.items():
            vectors = encode(test_texts)
            if seed == seeds[0]:
                figures[model] = vectors
            for kind in LABEL_KINDS:
                y = labels(split.test, kind)
                clusters = assign_clusters(vectors, len(set(y)), seed)
                rows.append(ScoreRow(model, seed, kind, score(vectors, y, clusters)))
                if seed == seeds[0] and kind == "topic":
                    topic_clusters[model] = clusters
            queries = encode([q.question for q, _ in retrieval])
            mrr[model].append(
                mean_reciprocal_rank(queries, vectors, [hits for _, hits in retrieval])
            )
    return StudyResults(
        split=split,
        rows=rows,
        retrieval_mrr=mrr,
        retrieval_questions=[q.id for q, _ in retrieval],
        epoch_losses=losses,
        test_embeddings=figures,
        topic_clusters=topic_clusters,
    )


def summarise(results: StudyResults, kind: str) -> dict[str, dict[str, tuple[float, float]]]:
    """``{metric: {model: (mean, std)}}`` over seeds, for one labelling."""
    metrics = list(ClusteringScores.__dataclass_fields__)
    table: dict[str, dict[str, tuple[float, float]]] = {}
    for metric in metrics:
        table[metric] = {}
        for model in MODELS:
            values = [
                getattr(r.scores, metric)
                for r in results.rows
                if r.model == model and r.labels == kind
            ]
            table[metric][model] = (float(np.mean(values)), float(np.std(values)))
    return table


def neighbour_check(results: StudyResults) -> list[dict]:
    """For the first held-out chunk of each filing, its nearest held-out neighbours per model."""
    test = results.split.test
    probes = [next(i for i, c in enumerate(test) if c.doc_id == doc_id) for doc_id in CATALOG]
    report = []
    for probe in probes:
        entry = {"probe": _describe(test[probe]), "neighbours": {}}
        for model, vectors in results.test_embeddings.items():
            similarity = vectors @ vectors[probe]
            similarity[probe] = -np.inf
            entry["neighbours"][model] = [
                {**_describe(test[i]), "cosine": round(float(similarity[i]), 3)}
                for i in np.argsort(-similarity)[:PROBE_NEIGHBOURS]
            ]
        report.append(entry)
    return report


def write_outputs(results: StudyResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    test = results.split.test
    summaries = {kind: summarise(results, kind) for kind in LABEL_KINDS}
    neighbours = neighbour_check(results)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "split": {"train": len(results.split.train), "test": len(test)},
                "rows": [{**asdict(r), "scores": asdict(r.scores)} for r in results.rows],
                "summary": summaries,
                "retrieval_mrr": results.retrieval_mrr,
                "retrieval_questions": results.retrieval_questions,
                "epoch_losses": results.epoch_losses,
                "neighbours": neighbours,
            },
            indent=2,
        )
    )
    (out_dir / "summary.md").write_text(_summary_markdown(results, summaries, neighbours))
    topic = summaries["topic"]
    metric_bars(
        {m: topic[m] for m in ("silhouette", "ari", "nmi", "knn_purity", "similarity_gap")},
        out_dir / "metrics_topic.png",
    )
    filings = labels(test, "doc")
    tsne_maps(results.test_embeddings, labels(test, "topic"), filings, 0, out_dir / "tsne.png")
    similarity_heatmaps(results.test_embeddings, filings, out_dir / "similarity.png")


def _describe(chunk: Chunk) -> dict:
    return {
        "filing": CATALOG[chunk.doc_id].short_name,
        "pages": chunk.pages,
        "text": chunk.text[:90],
    }


def _summary_markdown(results: StudyResults, summaries: dict, neighbours: list[dict]) -> str:
    lines = [f"Held-out chunks: {len(results.split.test)} (train: {len(results.split.train)})", ""]
    for kind, table in summaries.items():
        lines += [
            f"### Labels: {kind}",
            "",
            "| metric | " + " | ".join(MODELS) + " |",
            "|---|" + "---|" * len(MODELS),
        ]
        for metric, by_model in table.items():
            cells = [f"{mean:.3f} ± {std:.3f}" for mean, std in (by_model[m] for m in MODELS)]
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        f"### Retrieval MRR: {len(results.retrieval_questions)} eval questions answered by a "
        f"held-out passage, ranked against the {len(results.split.test)} held-out passages",
        "",
        "| " + " | ".join(MODELS) + " |",
        "|" + "---|" * len(MODELS),
    ]
    lines.append(
        "| "
        + " | ".join(
            f"{np.mean(results.retrieval_mrr[m]):.3f} ± {np.std(results.retrieval_mrr[m]):.3f}"
            for m in MODELS
        )
        + " |"
    )
    lines += ["", "Questions: " + ", ".join(results.retrieval_questions), ""]
    lines += ["### Domain model training loss (mean per epoch)", ""]
    for seed, epoch_losses in results.epoch_losses.items():
        lines.append(f"- seed {seed}: " + ", ".join(f"{loss:.3f}" for loss in epoch_losses))
    topics = labels(results.split.test, "topic")
    lines += ["", "### KMeans (k = 3) cluster composition by topic (seed 0)", ""]
    for model, assignment in results.topic_clusters.items():
        clusters = cluster_composition(topics, assignment)
        lines.append(
            f"- {model}: "
            + "; ".join(
                f"cluster {i}: " + ", ".join(f"{t} {n}" for t, n in sorted(c.items()))
                for i, c in enumerate(clusters)
            )
        )
    lines += ["", "### Nearest held-out neighbours (seed 0)", ""]
    for entry in neighbours:
        p = entry["probe"]
        lines.append(f"**{p['filing']} {p['pages']}**: {p['text']}…")
        for model, hits in entry["neighbours"].items():
            same = sum(h["filing"] == p["filing"] for h in hits)
            listed = ", ".join(f"{h['filing']} {h['pages']} ({h['cosine']})" for h in hits)
            lines.append(f"- {model} ({same}/{len(hits)} same filing): {listed}")
        lines.append("")
    return "\n".join(lines)
