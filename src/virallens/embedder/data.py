"""Q2 preprocessing: a held-out split of the chunked filings and the self-supervised training pairs.

The corpus is the shared chunker's output (``virallens.common``), so Q1 and Q2 see the same text.
The catalogue labels (topic, document type, filing) are read only for evaluation. Document structure
does shape training, though: section-neighbour pairs come from one section of one filing, so filing
identity, and with it topic, decides which pairs are positives.
"""

import random
import re
from dataclasses import dataclass

from virallens.common.catalog import CATALOG
from virallens.common.chunking import Chunk

# Pages go to train/test in contiguous blocks, one block in TEST_EVERY held out (~25% of each
# filing). 3-page blocks put pages 10-12 in the first test block, so even the 11-page Macquarie
# opinion has held-out pages.
PAGE_BLOCK = 3
TEST_EVERY = 4
# A verbatim run this long (one to two sentences) is copied text, not shared legal phrasing.
SHARED_SPAN_CHARS = 100
# Inverse Cloze Task (Lee, Chang & Toutanova, ACL 2019): a sentence is the pseudo-query for the rest
# of its passage. Sentences shorter than this are citation fragments ("Ibid.") with no content.
ICT_MIN_SENTENCE_CHARS = 60
_SENTENCE = re.compile(r"(?<=[.?!”])\s+(?=[A-Z“])")


@dataclass(frozen=True)
class Split:
    train: list[Chunk]
    test: list[Chunk]


def split_by_page_blocks(chunks: list[Chunk]) -> Split:
    """Hold out every TEST_EVERY-th block of PAGE_BLOCK pages per filing.

    A test chunk sharing any SHARED_SPAN_CHARS-character span with a training chunk is dropped:
    the model would have seen it. This catches chunk-boundary overlap (CHUNK_OVERLAP characters), a
    counsel block the complaint repeats on pp. 48 and 51, Macquarie's syllabus restating its opinion
    verbatim, and recurring boilerplate (a 25%-of-total-assets clause, a §10(b) citation string).
    """
    train, held_out = [], []
    for chunk in chunks:
        is_test = (chunk.page_start - 1) // PAGE_BLOCK % TEST_EVERY == TEST_EVERY - 1
        (held_out if is_test else train).append(chunk)
    seen = set().union(*(span_windows(chunk.text) for chunk in train))
    test = [chunk for chunk in held_out if not span_windows(chunk.text) & seen]
    return Split(train=train, test=test)


def span_windows(text: str) -> set[str]:
    """Every SHARED_SPAN_CHARS-character substring of ``text``, at every offset."""
    return {text[i : i + SHARED_SPAN_CHARS] for i in range(len(text) - SHARED_SPAN_CHARS + 1)}


def labels(chunks: list[Chunk], kind: str) -> list[str]:
    """Evaluation labels: ``topic``, ``doc_type`` or ``doc`` (the filing itself)."""
    if kind == "doc":
        return [CATALOG[c.doc_id].short_name for c in chunks]
    if kind in ("topic", "doc_type"):
        return [str(getattr(CATALOG[c.doc_id], kind)) for c in chunks]
    raise ValueError(f"unknown label kind {kind!r}; expected topic, doc_type or doc")


def ict_pairs(chunks: list[Chunk], rng: random.Random) -> list[tuple[str, str]]:
    """One (sentence, passage-without-that-sentence) pair per chunk with a usable sentence."""
    pairs = []
    for chunk in chunks:
        sentences = _SENTENCE.split(chunk.text)
        candidates = [i for i, s in enumerate(sentences) if len(s) >= ICT_MIN_SENTENCE_CHARS]
        if len(sentences) < 2 or not candidates:
            continue
        i = rng.choice(candidates)
        rest = " ".join(sentences[:i] + sentences[i + 1 :])
        pairs.append((sentences[i], rest))
    return pairs


def section_neighbour_pairs(chunks: list[Chunk]) -> list[tuple[str, str]]:
    """Chunks two apart in their filing and in the same section: same argument, no shared overlap.

    Distance is measured in the original chunk order, so a pair never straddles a held-out block.
    """
    by_position = {(c.doc_id, c.index): c for c in chunks}
    pairs = []
    for chunk in chunks:
        partner = by_position.get((chunk.doc_id, chunk.index + 2))
        if partner is not None and partner.section == chunk.section:
            pairs.append((chunk.text, partner.text))
    return pairs
