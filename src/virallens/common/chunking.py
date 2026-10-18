"""Section-aware chunking of cleaned lines.

Filings are split at their headings first, so no chunk straddles two detected headings:

- level 0: ALL-CAPS headings (briefs, complaint) and Roman numeral headings (the two opinions);
- level 1 and 2: Title Case point headings that open a paragraph, lettered ("A. Risk Disclosures
  Under Item 105 Make No Implied") and numbered ("1. The Ninth Circuit's Approach Ignores The
  Actual"), possibly wrapped over several lines.

A section is labelled with its heading path ("ARGUMENT ... > C. ... > 1. ..."). Sentence-case
point headings ("1. An incomplete risk-factor statement is a quintessential half-truth") cannot be
told apart from a numbered paragraph and stay in the body. Each section is reflowed into
paragraphs and then split recursively at paragraph, sentence and clause boundaries. Every chunk
records its section and the pages it spans so answers can cite "p. 18, ARGUMENT".
"""

import bisect
import re
import statistics
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from virallens.common.cleaning import Line

# ~1000 characters is ~200-250 word-piece tokens of legal English, under the 256-token input
# limit of all-MiniLM-L6-v2 (the Q2 baseline) and well under bge-small's 512 (Q1 retrieval).
CHUNK_SIZE = 1000
# One to two sentences, so a sentence cut at a chunk boundary appears whole in one neighbour.
CHUNK_OVERLAP = 150
SEPARATORS = ["\n\n", ". ", "; ", ", ", " ", ""]
# Page 1 of every filing is a caption, cover or syllabus whose ALL-CAPS party and counsel lines are
# not headings; it opens the document's first section under this name.
FRONT_MATTER_SECTION = "Front matter"
# A "heading" followed by less body text than this is folded, heading text included, into the
# preceding section. Measured on data/ before folding: name-only signature lines and caption
# fragments ("MEGAN BARBERO" / "General Counsel") carry <= 58 characters and parent headings whose
# body is all subsections carry 0; the shortest real sections (the briefs' CONCLUSIONs) carry
# 80-81. Longer signature and address blocks (67-126) overlap that range, so no threshold separates
# them and they remain small sections of their own.
MIN_SECTION_CHARS = 64
# Justified body lines run close to the document's median width; a line ending a sentence and
# shorter than this share of the median ends a paragraph.
PARAGRAPH_END_MAX_WIDTH = 0.8
HEADING_MAX_CHARS = 120
HEADING_MIN_LETTERS = 4
HEADING_MIN_UPPER_SHARE = 0.9
# Lettered point headings follow Chicago-style Title Case, which leaves function words lowercase
# ("C. Claims under Rule 10b-5(a) and (c)"), so they are scored on the share of capitalised words
# of 4+ letters. Measured on data/ over lettered lines that open a paragraph: real headings score
# >= 0.67, sentence-case openings ("D. Unlike the Ninth Circuit, some courts") <= 0.60. The margin
# is narrow; a miss only mislabels a section, it never drops text.
TITLE_WORD_MIN_LETTERS = 4
LETTERED_TITLE_MIN_SHARE = 0.65
# Numbered point headings in data/ capitalise every word ("1. The Ninth Circuit's Approach Ignores
# The Actual"), which separates them from proper-noun-heavy numbered paragraphs ("1. Under the
# Securities Act of 1933").

_ROMAN_HEADING = re.compile(r"[IVX]+(?:\.\s+[A-Z].*)?")
_NUMBERED_PARAGRAPH = re.compile(r"\d{1,3}\. [A-Z]")
_SUBHEADING = re.compile(r"(?:(?P<letter>[A-H])|\d{1,2})\. (?P<rest>[A-Z].*)")
_TITLE_WORD = re.compile(r"[A-Za-z][A-Za-z’'-]*")
# A sentence end may carry a footnote marker ("Ibid.1").
_SENTENCE_END = re.compile(r"[.:;?!\"”')]\d{0,2}$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str
    page_start: int
    page_end: int


@dataclass(frozen=True)
class _Section:
    path: list[tuple[int, str]]  # (level, heading), outermost first
    heading_lines: list[Line]
    lines: list[Line]

    @property
    def heading(self) -> str:
        return " > ".join(text for _, text in self.path)

    @property
    def heading_open(self) -> bool:
        return bool(self.heading_lines) and not self.lines

    def extend_heading(self, line: Line) -> "_Section":
        *outer, (level, text) = self.path
        return _Section([*outer, (level, f"{text} {line.text}")], [*self.heading_lines, line], [])


def chunk_document(doc_id: str, lines: list[Line]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator="end",
        add_start_index=True,
    )
    median_width = statistics.median(len(line.text) for line in lines)
    chunks: list[Chunk] = []
    for section in _sections(lines):
        text, line_starts = _reflow(section.lines, median_width)
        pages = [line.page for line in section.lines]
        for document in splitter.create_documents([text]):
            start = document.metadata["start_index"]
            if start < 0:
                raise ValueError(f"{doc_id}: chunk not found in section {section.heading!r}")
            end = start + len(document.page_content) - 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{len(chunks):04d}",
                    doc_id=doc_id,
                    text=document.page_content,
                    section=section.heading,
                    page_start=pages[bisect.bisect_right(line_starts, start) - 1],
                    page_end=pages[bisect.bisect_right(line_starts, end) - 1],
                )
            )
    return chunks


def is_heading(text: str) -> bool:
    if _ROMAN_HEADING.fullmatch(text):
        return True
    letters = [c for c in text if c.isalpha()]
    return (
        len(text) <= HEADING_MAX_CHARS
        and len(letters) >= HEADING_MIN_LETTERS
        and sum(c.isupper() for c in letters) / len(letters) >= HEADING_MIN_UPPER_SHARE
    )


def _heading_level(text: str, opens_paragraph: bool) -> int | None:
    if is_heading(text):
        return 0
    match = _SUBHEADING.fullmatch(text)
    if not (match and opens_paragraph):
        return None
    if match["letter"]:
        return 1 if _is_lettered_title(match["rest"]) else None
    words = _TITLE_WORD.findall(match["rest"])
    return 2 if all(word[0].isupper() for word in words) else None


def _is_lettered_title(text: str) -> bool:
    words = [w for w in _TITLE_WORD.findall(text) if len(w) >= TITLE_WORD_MIN_LETTERS]
    return (
        bool(words) and sum(w[0].isupper() for w in words) / len(words) >= LETTERED_TITLE_MIN_SHARE
    )


def _sections(lines: list[Line]) -> list[_Section]:
    sections = [_Section([(0, FRONT_MATTER_SECTION)], [], [])]
    wrap_width = PARAGRAPH_END_MAX_WIDTH * statistics.median(len(line.text) for line in lines)
    previous = ""
    for line in lines:
        current = sections[-1]
        opens_paragraph = current.heading_open or bool(_SENTENCE_END.search(previous))
        level = _heading_level(line.text, opens_paragraph) if line.page > 1 else None
        open_level = current.path[-1][0] if current.heading_open else None
        wrapped_subheading = (
            level is None
            and bool(open_level)
            and len(previous) >= wrap_width
            and _is_lettered_title(line.text)
        )
        if (level == 0 and open_level == 0) or wrapped_subheading:
            sections[-1] = current.extend_heading(line)
        elif level is not None:
            outer = [entry for entry in current.path if entry[0] < level]
            sections.append(_Section([*outer, (level, line.text)], [line], []))
        else:
            current.lines.append(line)
        previous = line.text
    return _fold_short_sections(sections)


def _fold_short_sections(sections: list[_Section]) -> list[_Section]:
    folded: list[_Section] = []
    for section in sections:
        if folded and len(" ".join(line.text for line in section.lines)) < MIN_SECTION_CHARS:
            previous = folded.pop()
            lines = [*previous.lines, *section.heading_lines, *section.lines]
            folded.append(_Section(previous.path, previous.heading_lines, lines))
        else:
            folded.append(section)
    return folded


def _reflow(lines: list[Line], median_width: float) -> tuple[str, list[int]]:
    """Join a section's lines into paragraphs; also return each line's offset in the result."""
    parts: list[str] = []
    line_starts: list[int] = []
    offset = 0
    for i, line in enumerate(lines):
        if i:
            separator = (
                "\n\n" if _ends_paragraph(lines[i - 1].text, line.text, median_width) else " "
            )
            parts.append(separator)
            offset += len(separator)
        line_starts.append(offset)
        parts.append(line.text)
        offset += len(line.text)
    return "".join(parts), line_starts


def _ends_paragraph(previous: str, current: str, median_width: float) -> bool:
    # A sentence-case point heading ("1. An incomplete risk-factor statement is a quintessential
    # half-truth") has no closing punctuation, so it stays attached to the paragraph it introduces.
    if not _SENTENCE_END.search(previous):
        return False
    return (
        bool(_NUMBERED_PARAGRAPH.match(current))
        or len(previous) < PARAGRAPH_END_MAX_WIDTH * median_width
    )
