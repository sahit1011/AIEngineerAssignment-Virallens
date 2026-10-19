"""Q2 training: adapt a small general-purpose sentence encoder to the filings, self-supervised.

The baseline ``all-MiniLM-L6-v2`` is fine-tuned with MultipleNegativesRankingLoss (in-batch
negatives) on Inverse Cloze Task pairs and same-section neighbour pairs drawn from the training
split only, and a linear head projects its 384-d output to 128-d. No labels are used.

The loop is written out rather than run through ``SentenceTransformerTrainer``, which needs the
``datasets`` and ``accelerate`` packages (sentence-transformers' "train" extra) for one small job.
"""

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.modules import Dense

from virallens.common.chunking import Chunk
from virallens.embedder.data import ict_pairs, section_neighbour_pairs

logger = logging.getLogger(__name__)

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The brief asks for no long embeddings; 128-d is a third of the baseline's 384.
DOMAIN_DIM = 128
# Untuned, and small because the training split is only ~280 chunks (~26 steps per epoch). The loss
# per epoch is recorded in the study's metrics.json.
EPOCHS = 4
BATCH_SIZE = 16
# The encoder starts from sentence-transformers' default fine-tuning rate (``fit``: AdamW, 2e-5).
# The projection head starts from random weights, so it gets a larger, untuned rate.
ENCODER_LR = 2e-5
HEAD_LR = 1e-3
# Linear warm-up to the full rate over the first 10% of steps, then linear decay to zero over the
# rest: the shape of ``fit``'s "WarmupLinear" schedule.
WARMUP_SHARE = 0.1
# CPU keeps runs bit-reproducible per seed; MPS kernels are not deterministic.
DEVICE = "cpu"


def load_base(cache_dir: Path) -> SentenceTransformer:
    return SentenceTransformer(BASE_MODEL, device=DEVICE, cache_folder=str(cache_dir))


@dataclass(frozen=True)
class TrainedModel:
    model: SentenceTransformer
    epoch_losses: list[float]


def train_domain_model(train: list[Chunk], seed: int, cache_dir: Path) -> TrainedModel:
    _seed_everything(seed)
    model = load_base(cache_dir)
    width = model.get_embedding_dimension()
    if width is None:
        raise ValueError(f"{BASE_MODEL} does not report its embedding dimension")
    head = Dense(
        in_features=width,
        out_features=DOMAIN_DIM,
        activation_function=torch.nn.Identity(),
    )
    model.append(head)
    model.to(DEVICE)
    loss = MultipleNegativesRankingLoss(model)
    head_ids = {id(p) for p in head.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": ENCODER_LR},
            {"params": list(head.parameters()), "lr": HEAD_LR},
        ]
    )
    rng = random.Random(seed)
    neighbours = section_neighbour_pairs(train)
    # ict_pairs yields one pair per eligible chunk whatever sentence it samples, so the count holds
    # for every epoch.
    pairs_per_epoch = len(ict_pairs(train, random.Random(seed))) + len(neighbours)
    total = pairs_per_epoch // BATCH_SIZE * EPOCHS
    warmup = max(1, int(WARMUP_SHARE * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear(warmup, total))
    model.train()
    epoch_losses = []
    for epoch in range(EPOCHS):
        # A fresh ICT sentence per chunk each epoch: the same passage, a different pseudo-query.
        pairs = ict_pairs(train, rng) + neighbours
        rng.shuffle(pairs)
        epoch_loss = []
        for start in range(0, len(pairs) - BATCH_SIZE + 1, BATCH_SIZE):
            anchors, positives = zip(*pairs[start : start + BATCH_SIZE], strict=True)
            # Inputs are built on the CPU, which is also the training device.
            features = [model.preprocess(list(texts)) for texts in (anchors, positives)]
            # MNRL takes the positive for anchor i to be candidate i; `labels` is unused by it.
            value = loss(features, labels=None)
            optimizer.zero_grad()
            value.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss.append(value.item())
        epoch_losses.append(float(np.mean(epoch_loss)))
        logger.info("seed %d epoch %d: mean loss %.4f", seed, epoch + 1, epoch_losses[-1])
    model.eval()
    return TrainedModel(model=model, epoch_losses=epoch_losses)


def warmup_linear(warmup: int, total: int) -> Callable[[int], float]:
    """Learning-rate multiplier: 0 -> 1 over ``warmup`` steps, then 1 -> 0 at ``total``."""

    def multiplier(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, (total - step) / (total - warmup))

    return multiplier


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
