"""Metadata for the five filings in ``data/``, transcribed from each document's cover page.

The PDFs carry no usable embedded metadata, so this table is the single source of truth.
``topic`` groups filings by legal subject matter; it is used as a reference labelling in
the Q2 clustering study and must never be fed into training.
"""

from dataclasses import dataclass
from enum import StrEnum


class DocType(StrEnum):
    OPINION = "opinion"
    BRIEF = "brief"
    COMPLAINT = "complaint"


class Topic(StrEnum):
    DISCLOSURE_OMISSIONS = "disclosure_omissions"
    SPECIALIST_EXECUTION = "specialist_execution"
    FUND_CONCENTRATION = "fund_concentration"


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    title: str
    case: str
    court: str
    doc_type: DocType
    topic: Topic


CATALOG: dict[str, DocumentInfo] = {
    info.doc_id: info
    for info in [
        DocumentInfo(
            doc_id="22-1165_10n2",
            title="Macquarie Infrastructure Corp. v. Moab Partners, L.P. (slip opinion)",
            case="Macquarie v. Moab, No. 22-1165",
            court="U.S. Supreme Court (decided Apr. 12, 2024)",
            doc_type=DocType.OPINION,
            topic=Topic.DISCLOSURE_OMISSIONS,
        ),
        DocumentInfo(
            doc_id="20240809133545974_23-980_petbrief",
            title="Facebook, Inc. v. Amalgamated Bank: Brief for the Petitioners",
            case="Facebook v. Amalgamated Bank, No. 23-980",
            court="U.S. Supreme Court",
            doc_type=DocType.BRIEF,
            topic=Topic.DISCLOSURE_OMISSIONS,
        ),
        DocumentInfo(
            doc_id="23-980bsacus_facebookvamalgamatedbank",
            title=(
                "Facebook, Inc. v. Amalgamated Bank: Brief for the United States "
                "as Amicus Curiae Supporting Respondents"
            ),
            case="Facebook v. Amalgamated Bank, No. 23-980",
            court="U.S. Supreme Court",
            doc_type=DocType.BRIEF,
            topic=Topic.DISCLOSURE_OMISSIONS,
        ),
        DocumentInfo(
            doc_id="USCOURTS-ilnd-1_04-cv-00397-5",
            title=(
                "Last Atlantis Capital LLC v. AGS Specialist Partners: Memorandum Opinion and Order"
            ),
            case="Last Atlantis v. AGS, No. 04 C 397",
            court="U.S. District Court, N.D. Illinois (Mar. 26, 2010)",
            doc_type=DocType.OPINION,
            topic=Topic.SPECIALIST_EXECUTION,
        ),
        DocumentInfo(
            doc_id="comp26286",
            title=(
                "SEC v. David Yow Shang Chiueh and Upright Financial Corp.: First Amended Complaint"
            ),
            case="SEC v. Chiueh, No. 25 Civ. 1920",
            court="U.S. District Court, D. New Jersey (filed Apr. 11, 2025)",
            doc_type=DocType.COMPLAINT,
            topic=Topic.FUND_CONCENTRATION,
        ),
    ]
}
