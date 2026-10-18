"""Load the filings in a data directory into cleaned, section-aware chunks."""

from pathlib import Path

from pypdf import PdfReader

from virallens.common.catalog import CATALOG
from virallens.common.chunking import Chunk, chunk_document
from virallens.common.cleaning import clean_document


def load_corpus(data_dir: Path) -> list[Chunk]:
    pdfs = sorted(data_dir.glob("*.pdf"))
    found = {pdf.stem for pdf in pdfs}
    if found != CATALOG.keys():
        raise ValueError(
            f"{data_dir}: PDFs do not match the catalog; "
            f"unlisted={sorted(found - CATALOG.keys())}, missing={sorted(CATALOG.keys() - found)}"
        )
    chunks: list[Chunk] = []
    for pdf in pdfs:
        lines = clean_document([page.extract_text() for page in PdfReader(pdf).pages])
        if not lines:
            raise ValueError(f"{pdf}: no text left after cleaning")
        chunks.extend(chunk_document(pdf.stem, lines))
    return chunks
