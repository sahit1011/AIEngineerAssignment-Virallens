"""How passages are shown to the models and how answers cite them.

A passage is cited by a short key, ``<filing short name>:<chunk index>`` (``[macquarie:29]``),
because raw chunk ids are too long for a model to copy reliably.
"""

import re

from virallens.common.catalog import CATALOG
from virallens.common.chunking import Chunk

_KEY = re.compile(r"[a-z][a-z-]*:\d+")
# One bracket may hold several keys: "[macquarie:4]" or "[macquarie:4; fb-petitioners:6]".
_BRACKET = re.compile(r"\[([^\[\]]+)\]")
# gpt-oss writes its native citation brackets ("【macquarie:4】", "【macquarie:4†L1-L3】") whatever
# the prompt asks.
_NATIVE_CITATION = re.compile(r"【([^】†]+)(?:†[^】]*)?】")


def citation_key(chunk: Chunk) -> str:
    return f"{CATALOG[chunk.doc_id].short_name}:{chunk.index}"


def normalise_citations(text: str) -> str:
    """Rewrite native ``【key】`` citations as ``[key]``."""
    return _NATIVE_CITATION.sub(r"[\1]", text)


def cited_keys(text: str) -> list[str]:
    """Every passage key cited in ``text``, in order of appearance."""
    return [
        part.strip()
        for group in _BRACKET.findall(text)
        for part in re.split(r"[,;]", group)
        if _KEY.fullmatch(part.strip())
    ]


def format_context(chunks: list[Chunk]) -> str:
    """Render chunks as citable blocks: ``[key] filing | heading | pages``, then the text."""
    blocks = []
    for chunk in chunks:
        heading = chunk.section.rsplit(" > ", 1)[-1]
        blocks.append(
            f"[{citation_key(chunk)}] {CATALOG[chunk.doc_id].title} | {heading} | "
            f"{chunk.pages}\n{chunk.text}"
        )
    return "\n\n".join(blocks)
