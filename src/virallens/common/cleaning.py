"""Turn raw per-page PDF text into clean, page-tagged lines.

Noise observed in pypdf output for ``data/*.pdf``:

- underline rules ("_________________") that push running heads off the page edge;
- page furniture at page edges: ECF stamps ("Case 2:25-cv-01920 ... Page 2 of 51 PageID: 67"),
  page numbers ("2", "(I)", "VI") and running heads ("Opinion of the Court");
- table-of-contents / table-of-authorities pages made of dot leaders;
- words broken across lines with a hyphen ("sec-" / "tion");
- a spurious space before an intra-word hyphen ("warned -of", "Form 10 -K");
- the same term typeset with an en dash or a hyphen ("Rule 10b–5" in 40 chunks, "10b-5" in 106).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

# Running heads reach 3 lines deep; ECF stamps and page numbers sit in the last 2.
EDGE_WINDOW = 3
# An edge line recurring on at least this many pages, and this share of a filing's pages, is
# furniture. Margins measured on data/: docs/DESIGN.md.
FURNITURE_MIN_PAGES = 3
FURNITURE_MIN_SHARE = 0.25
# Pages with at least this share of dot-leader lines are tables of contents or authorities.
INDEX_PAGE_MIN_SHARE = 0.3

_RULE_LINE = re.compile(r"_{3,}")
_PAGE_NUMBER = re.compile(r"\(?(?:\d+|[IVXLC]+)\)?")
_DIGITS = re.compile(r"\d+")
_DOT_LEADER = re.compile(r"\.{4,}")
_SPACE_BEFORE_HYPHEN = re.compile(r"(?<=\w) +-(?=\w)")
_EN_DASH_IN_WORD = re.compile(r"(?<=\w)–(?=\w)")
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")
_LEADING_WORD = re.compile(r"[a-z]+")
_TRAILING_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*$")
_LINE_END_DASHES = "-–—"


@dataclass(frozen=True)
class Line:
    page: int
    text: str


def clean_document(raw_pages: list[str]) -> list[Line]:
    """Clean one document. ``raw_pages[i]`` is the extracted text of page ``i + 1``."""
    pages = [
        [text for raw in page.splitlines() if (text := " ".join(raw.split()))] for page in raw_pages
    ]
    pages = [[text for text in page if not _RULE_LINE.fullmatch(text)] for page in pages]
    furniture = _furniture_keys(pages)
    lines = [
        Line(page=number, text=_normalise_dashes(text))
        for number, page in enumerate(pages, start=1)
        if not _is_index_page(page)
        for text in _strip_furniture(page, furniture)
    ]
    # Joining brings dashes that ended a line ("10b–" + "5(b)") next to word characters.
    return [Line(line.page, _normalise_dashes(line.text)) for line in _join_broken_lines(lines)]


def _normalise_dashes(text: str) -> str:
    # An en dash between word characters is either the same term spelled two ways or a range
    # ("109a–127a"); a hyphen reads the same for both and lets keyword search match either.
    return _EN_DASH_IN_WORD.sub("-", _SPACE_BEFORE_HYPHEN.sub("-", text))


def _furniture_key(text: str) -> str:
    if _PAGE_NUMBER.fullmatch(text):
        return "<page-number>"
    return _DIGITS.sub("#", text)


def _edge_positions(page: list[str]) -> set[int]:
    n = len(page)
    return set(range(min(EDGE_WINDOW, n))) | set(range(max(0, n - EDGE_WINDOW), n))


def _furniture_keys(pages: list[list[str]]) -> set[str]:
    pages_per_key = Counter(
        key for page in pages for key in {_furniture_key(page[i]) for i in _edge_positions(page)}
    )
    threshold = max(FURNITURE_MIN_PAGES, math.ceil(FURNITURE_MIN_SHARE * len(pages)))
    return {key for key, count in pages_per_key.items() if count >= threshold}


def _strip_furniture(page: list[str], furniture: set[str]) -> list[str]:
    edges = _edge_positions(page)
    return [
        text
        for i, text in enumerate(page)
        if not (i in edges and _furniture_key(text) in furniture)
    ]


def _is_index_page(page: list[str]) -> bool:
    if not page:
        return False
    leader_lines = sum(1 for text in page if _DOT_LEADER.search(text))
    return leader_lines / len(page) >= INDEX_PAGE_MIN_SHARE


def _join_broken_lines(lines: list[Line]) -> list[Line]:
    """Merge a line ending in a dash with the line that continues it.

    A lowercase syllable break ("sec-" + "tion") drops the hyphen unless the document itself
    spells the word hyphenated elsewhere ("risk-factor") and never solid, or the broken word is
    already a compound ("cut-and-" + "dry"). Any other trailing dash ("10b-" + "5(b)", an em dash)
    is kept and the lines are joined without a space.
    """
    vocabulary = Counter(word for line in lines for word in _WORD.findall(line.text.lower()))
    merged: list[Line] = []
    for line in lines:
        if merged and merged[-1].text[-1] in _LINE_END_DASHES:
            previous = merged.pop()
            merged.append(Line(previous.page, _join(previous.text, line.text, vocabulary)))
        else:
            merged.append(line)
    return merged


def _join(left: str, right: str, vocabulary: Counter[str]) -> str:
    dash, stem = left[-1], left[:-1].rstrip()
    left_word = _TRAILING_WORD.search(stem)
    right_word = _LEADING_WORD.match(right)
    if dash == "-" and left_word and right_word:
        left_token = left_word.group().lower()
        solid = left_token + right_word.group()
        hyphenated = f"{left_token}-{right_word.group()}"
        if "-" in left_token or (vocabulary[hyphenated] and not vocabulary[solid]):
            return f"{stem}-{right}"
        return stem + right
    return stem + dash + right
