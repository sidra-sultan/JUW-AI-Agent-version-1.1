"""Build the JUW FAISS knowledge index.

Features:
- Local PDF ingestion
- Official JUW webpage ingestion
- Official JUW policy PDF ingestion
- Resilient retries and per-source failure isolation
- Structure-aware chunking
- Heading/section preservation
- Paragraph and sentence-aware chunk boundaries
- Controlled overlap
- Duplicate chunk removal
- Stable chunk IDs
- Rich metadata for retrieval/debugging
- Refuses to overwrite an index when extraction completely fails
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = ROOT / "data" / "data"
INDEX_DIR = ROOT / "data" / "faiss_index"


# ============================================================
# OFFICIAL JUW SOURCES
# ============================================================

WEB_SOURCES = [
    "https://www.juw.edu.pk/juw-qec/annual-activity-calendar/",
    "https://www.juw.edu.pk/juw-qec/contact-us/",
    "https://www.juw.edu.pk/juw-qec/policies-evaluation/",
]


QEC_POLICY_PDFS = [
    "https://www.juw.edu.pk/resource/uploads/2022/08/Admission-Policy0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/Alumni-Coucil0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/AntiDrugTobaccoPolicy0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/Anti-HarassmentPolicy0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/CareerServicesPlacementOffice-0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/Evaluation-of-thesis.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/ExaminationPolicy0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/Examination-Policy-MS-M.Phil_.-PhD.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/FacultyHandBook0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/FeesPayment0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/GrievancePolicy0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/InformationSharingPolicy0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/PlagiarismPolicy0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/Publication-Policy-for-Degree-Completion.-MS-M.Phil_.-PhD.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/RegistrationsEnrollment0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/SemesterSystem0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/StudentsDisabilityPolicy0822.pdf",
    "https://www.juw.edu.pk/resource/uploads/2022/08/Transparency-Policy.pdf",
    "https://www.juw.edu.pk/resource/uploads/2025/06/Student-Support-Services.pdf",
    "https://www.juw.edu.pk/resource/uploads/2025/06/Faculty-Training-and-Development.pdf",
]


# ============================================================
# STRUCTURED INFORMATION
# ============================================================

QEC_CONTACT_DETAILS = """JUW Quality Enhancement Cell (QEC) Contact Us

Email: info@juw.edu.pk
Address: 5-C Nazimabad, Karachi-74600, Pakistan
Phone: 3661-9902, 3662-0857-9
Fax: 3662-0614
"""


# ============================================================
# CONFIGURATION
# ============================================================

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.5"))

MIN_TEXT_CHARS = int(os.getenv("MIN_TEXT_CHARS", "80"))

# Target chunk size.
# This is deliberately moderate for university information.
TARGET_CHUNK_SIZE = int(os.getenv("TARGET_CHUNK_SIZE", "900"))

# Hard upper limit.
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1200"))

# Approximate overlap between neighboring chunks.
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Ignore extremely tiny chunks.
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "120"))


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("juw-indexer")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""

    if not text:
        return ""

    # Normalize different newline formats.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize non-breaking spaces.
    text = text.replace("\xa0", " ")

    # Remove excessive spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_for_hash(text: str) -> str:
    """Create a stable representation for duplicate detection."""

    text = normalize_text(text).lower()

    # Remove insignificant whitespace differences.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def document_hash(text: str) -> str:
    """SHA-256 hash of normalized text."""

    normalized = normalize_for_hash(text)

    return hashlib.sha256(
        normalized.encode("utf-8", "ignore")
    ).hexdigest()


# ============================================================
# DOCUMENT TYPE
# ============================================================

def document_type(source: str) -> str:
    lower = source.lower()

    if lower.startswith("http") and lower.endswith(".pdf"):
        return "web_pdf"

    if lower.startswith("http"):
        return "webpage"

    return "pdf"


# ============================================================
# HEADING DETECTION
# ============================================================

def looks_like_heading(line: str) -> bool:
    """Determine whether a line is probably a heading."""

    line = " ".join(line.split()).strip()

    if not line:
        return False

    if len(line) > 120:
        return False

    words = line.split()

    if len(words) > 15:
        return False

    # Common academic/document headings.
    if re.match(
        r"^(chapter|section|part|appendix|article|policy|procedure|"
        r"objective|objectives|introduction|scope|purpose|"
        r"eligibility|requirements|contact|references)\b",
        line,
        re.IGNORECASE,
    ):
        return True

    # Numbered headings:
    # 1.
    # 1.1
    # 2.3.4
    if re.match(r"^\d+(\.\d+)*[\s.)-]+", line):
        return True

    # Roman numeral headings.
    if re.match(r"^[IVXLC]+[\s.)-]+", line):
        return True

    # All-uppercase short lines.
    if line.isupper() and len(words) <= 12:
        return True

    # Lines ending with colon can represent labels.
    if line.endswith(":") and len(words) <= 12:
        return True

    return False


def infer_heading(text: str) -> str | None:
    """Find the first strong heading candidate."""

    for raw_line in text.splitlines():

        line = " ".join(raw_line.split()).strip()

        if not line:
            continue

        if looks_like_heading(line):
            return line.rstrip(":").strip()

    return None


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while avoiding extremely aggressive splitting."""

    text = normalize_text(text)

    if not text:
        return []

    # Protect common abbreviations.
    protected = text

    abbreviations = [
        "Dr.",
        "Mr.",
        "Mrs.",
        "Ms.",
        "Prof.",
        "No.",
        "e.g.",
        "i.e.",
        "etc.",
        "Ph.D.",
        "M.Phil.",
    ]

    placeholders = {}

    for index, abbreviation in enumerate(abbreviations):
        placeholder = f"__ABBR_{index}__"
        placeholders[placeholder] = abbreviation
        protected = protected.replace(abbreviation, placeholder)

    # Sentence boundary:
    # punctuation followed by whitespace and capital/number.
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        protected,
    )

    sentences = []

    for part in parts:

        for placeholder, abbreviation in placeholders.items():
            part = part.replace(placeholder, abbreviation)

        part = part.strip()

        if part:
            sentences.append(part)

    return sentences


# ============================================================
# PARAGRAPH / SECTION EXTRACTION
# ============================================================

def extract_blocks(text: str) -> list[dict[str, str]]:
    """
    Convert a document into semantic blocks.

    Each block contains:
    - heading
    - content
    """

    text = normalize_text(text)

    if not text:
        return []

    lines = text.splitlines()

    blocks = []

    current_heading = None
    current_lines = []

    def flush_block():
        nonlocal current_lines

        content = "\n".join(current_lines).strip()

        if content:

            blocks.append(
                {
                    "heading": current_heading or "",
                    "content": content,
                }
            )

        current_lines = []

    for raw_line in lines:

        line = " ".join(raw_line.split()).strip()

        if not line:
            # Blank line means possible paragraph boundary.
            if current_lines:
                flush_block()

            continue

        if looks_like_heading(line):

            # Finish previous block.
            if current_lines:
                flush_block()

            current_heading = line.rstrip(":").strip()

            continue

        current_lines.append(line)

    if current_lines:
        flush_block()

    return blocks


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

def build_semantic_chunks(
    text: str,
    target_size: int = TARGET_CHUNK_SIZE,
    max_size: int = MAX_CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    min_size: int = MIN_CHUNK_SIZE,
) -> list[dict[str, str]]:
    """
    Build chunks using document structure.

    Priority:

    1. Section/heading boundaries
    2. Paragraph boundaries
    3. Sentence boundaries
    4. Hard character limit

    This prevents arbitrary cuts in the middle of important information.
    """

    blocks = extract_blocks(text)

    if not blocks:
        return []

    chunks = []

    current_text = ""
    current_heading = ""

    def add_current():

        nonlocal current_text

        cleaned = normalize_text(current_text)

        if not cleaned:
            current_text = ""
            return

        chunks.append(
            {
                "heading": current_heading,
                "content": cleaned,
            }
        )

        current_text = ""

    for block in blocks:

        heading = block["heading"]
        content = normalize_text(block["content"])

        if not content:
            continue

        # If the heading changes, prefer a new chunk.
        if (
            current_text
            and heading
            and current_heading
            and heading != current_heading
        ):
            add_current()

        if heading:
            current_heading = heading

        # If the whole block fits, add it naturally.
        proposed = (
            f"{current_text}\n\n{content}"
            if current_text
            else content
        )

        if len(proposed) <= target_size:
            current_text = proposed
            continue

        # If current chunk is already meaningful, close it.
        if current_text:
            add_current()

        # Now split the oversized block by sentences.
        sentences = split_into_sentences(content)

        if not sentences:
            sentences = [content]

        sentence_buffer = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            proposed = (
                f"{sentence_buffer} {sentence}"
                if sentence_buffer
                else sentence
            )

            # Keep normal chunks around target size.
            if len(proposed) <= target_size:
                sentence_buffer = proposed
                continue

            # Flush existing sentence buffer.
            if sentence_buffer:
                chunks.append(
                    {
                        "heading": current_heading,
                        "content": sentence_buffer.strip(),
                    }
                )

            # If a single sentence is huge, hard split it.
            if len(sentence) > max_size:

                start = 0

                while start < len(sentence):

                    end = min(
                        start + max_size,
                        len(sentence),
                    )

                    piece = sentence[start:end].strip()

                    if piece:
                        chunks.append(
                            {
                                "heading": current_heading,
                                "content": piece,
                            }
                        )

                    if end >= len(sentence):
                        break

                    start = max(
                        end - overlap,
                        start + 1,
                    )

                sentence_buffer = ""

            else:
                sentence_buffer = sentence

        if sentence_buffer:
            current_text = sentence_buffer

    if current_text:
        add_current()

    # ========================================================
    # MERGE TINY CHUNKS
    # ========================================================

    merged = []

    for chunk in chunks:

        content = chunk["content"]
        heading = chunk["heading"]

        if (
            merged
            and len(content) < min_size
            and len(
                merged[-1]["content"]
                + " "
                + content
            ) <= max_size
        ):

            merged[-1]["content"] = (
                merged[-1]["content"]
                + "\n\n"
                + content
            ).strip()

        else:

            merged.append(
                {
                    "heading": heading,
                    "content": content,
                }
            )

    # ========================================================
    # ADD LIGHT OVERLAP
    # ========================================================

    final_chunks = []

    for index, chunk in enumerate(merged):

        content = chunk["content"]
        heading = chunk["heading"]

        previous_tail = ""

        if index > 0:

            previous_content = merged[index - 1]["content"]

            if len(previous_content) > overlap:

                previous_tail = previous_content[-overlap:].strip()

        if previous_tail:

            content = (
                previous_tail
                + "\n\n"
                + content
            )

        final_chunks.append(
            {
                "heading": heading,
                "content": content.strip(),
            }
        )

    return final_chunks


# ============================================================
# LOCAL PDF LOADER
# ============================================================

def pdf_to_documents(path: Path) -> list[Document]:
    """Extract readable pages from a local PDF."""

    documents = []

    reader = PdfReader(
        str(path),
        strict=False,
    )

    if reader.is_encrypted:

        logger.warning(
            "PDF is encrypted/protected: %s",
            path.name,
        )

        try:
            reader.decrypt("")
        except Exception:
            pass

    for page_number, page in enumerate(reader.pages):

        try:

            text = (
                page.extract_text()
                or ""
            ).strip()

        except Exception as error:

            logger.error(
                "PDF page extraction failed: %s page %d: %s",
                path.name,
                page_number + 1,
                error,
            )

            continue

        if len(text) < MIN_TEXT_CHARS:

            logger.warning(
                "Low/no extractable text: %s page %d (%d chars). OCR not enabled.",
                path.name,
                page_number + 1,
                len(text),
            )

            continue

        text = normalize_text(text)

        heading = infer_heading(text)

        metadata = {
            "source": str(path),
            "document_name": path.name,
            "document_type": "pdf",
            "page": page_number,
        }

        if heading:
            metadata["heading"] = heading

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


# ============================================================
# HTTP FETCHING
# ============================================================

def fetch_with_retries(url: str) -> requests.Response:

    last_error = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; JUW-Information-Assistant/2.0)"
        )
    }

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=(
                    10,
                    REQUEST_TIMEOUT,
                ),
                allow_redirects=True,
            )

            response.raise_for_status()

            return response

        except requests.RequestException as error:

            last_error = error

            logger.warning(
                "Fetch failed (attempt %d/%d): %s",
                attempt,
                MAX_RETRIES,
                url,
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY * attempt
                )

    raise RuntimeError(
        f"Failed after {MAX_RETRIES} attempts: {url}"
    ) from last_error


# ============================================================
# WEBPAGE LOADER
# ============================================================

def web_page_to_document(url: str) -> Document:

    response = fetch_with_retries(url)

    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    if "text/html" not in content_type:

        raise ValueError(
            "Expected HTML but received "
            f"{content_type or 'unknown content type'}"
        )

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    # Remove irrelevant HTML.
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
            "form",
        ]
    ):
        tag.decompose()

    # Prefer the main content when available.
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(
            "div",
            class_=re.compile(
                r"(content|main|page)",
                re.IGNORECASE,
            ),
        )
    )

    source_element = main or soup

    text = "\n".join(
        line.strip()
        for line in source_element
        .get_text("\n")
        .splitlines()
        if line.strip()
    )

    text = normalize_text(text)

    if len(text) < MIN_TEXT_CHARS:

        raise ValueError(
            "Webpage contained too little extractable text"
        )

    title = (
        soup.title.get_text(
            " ",
            strip=True,
        )
        if soup.title
        else url
    )

    return Document(
        page_content=text,
        metadata={
            "source": url,
            "document_name": title,
            "document_type": "webpage",
            "url": url,
        },
    )


# ============================================================
# REMOTE PDF LOADER
# ============================================================

def web_pdf_to_documents(url: str) -> list[Document]:

    response = fetch_with_retries(url)

    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    if (
        "pdf" not in content_type
        and not url.lower().endswith(".pdf")
    ):

        raise ValueError(
            "Expected PDF but received "
            f"{content_type or 'unknown content type'}"
        )

    reader = PdfReader(
        BytesIO(response.content),
        strict=False,
    )

    documents = []

    for page_number, page in enumerate(reader.pages):

        try:

            text = (
                page.extract_text()
                or ""
            ).strip()

        except Exception as error:

            logger.error(
                "Remote PDF page failed: %s page %d: %s",
                url,
                page_number + 1,
                error,
            )

            continue

        if len(text) < MIN_TEXT_CHARS:

            logger.warning(
                "Remote PDF page has little/no text: %s page %d",
                url,
                page_number + 1,
            )

            continue

        text = normalize_text(text)

        heading = infer_heading(text)

        metadata = {
            "source": url,
            "document_name": Path(url).name,
            "document_type": "web_pdf",
            "page": page_number,
            "url": url,
        }

        if heading:
            metadata["heading"] = heading

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


# ============================================================
# STRUCTURED DOCUMENT CHUNKING
# ============================================================

def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Convert source documents into structure-aware chunks.

    Unlike naive fixed-size splitting, this attempts to preserve:
    - headings
    - paragraphs
    - sentences
    - page boundaries
    """

    chunks = []

    for document in documents:

        source_metadata = dict(
            document.metadata
        )

        semantic_chunks = build_semantic_chunks(
            document.page_content
        )

        for index, chunk in enumerate(
            semantic_chunks
        ):

            text = normalize_text(
                chunk["content"]
            )

            if len(text) < MIN_CHUNK_SIZE:
                continue

            metadata = dict(
                source_metadata
            )

            heading = (
                chunk.get("heading")
                or metadata.get("heading")
            )

            if heading:
                metadata["heading"] = heading

            metadata["chunk_index"] = index

            metadata["chunk_id"] = hashlib.sha1(
                (
                    f"{metadata.get('source')}"
                    f"|{metadata.get('page')}"
                    f"|{index}"
                    f"|{document_hash(text)}"
                ).encode(
                    "utf-8",
                    "ignore",
                )
            ).hexdigest()[:16]

            metadata["content_hash"] = (
                document_hash(text)
            )

            chunks.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

    return chunks


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_chunks(
    chunks: list[Document],
) -> list[Document]:

    unique = []

    seen = set()

    for chunk in chunks:

        content_hash = chunk.metadata.get(
            "content_hash"
        )

        if not content_hash:

            content_hash = document_hash(
                chunk.page_content
            )

        if content_hash in seen:
            continue

        seen.add(content_hash)

        unique.append(chunk)

    return unique


# ============================================================
# SAFE SOURCE LOADING
# ============================================================

def add_source_safely(
    documents: list[Document],
    loader,
    label: str,
):

    try:

        new_documents = loader()

        documents.extend(
            new_documents
        )

        logger.info(
            "Loaded %d documents: %s",
            len(new_documents),
            label,
        )

    except Exception:

        logger.exception(
            "Source failed and was skipped: %s",
            label,
        )


# ============================================================
# INDEX BUILDING
# ============================================================

def main():

    logger.info("=" * 70)
    logger.info("JUW FAISS KNOWLEDGE INDEX BUILD")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # LOCAL PDFs
    # --------------------------------------------------------

    pdf_files = sorted(
        PDF_DIR.glob("*.pdf")
    )

    if not pdf_files:

        logger.error(
            "No PDF files found in %s",
            PDF_DIR,
        )

        sys.exit(1)

    documents: list[Document] = []

    for pdf in pdf_files:

        try:

            logger.info(
                "Reading local PDF: %s",
                pdf.name,
            )

            before = len(documents)

            documents.extend(
                pdf_to_documents(pdf)
            )

            logger.info(
                "Loaded %d pages from %s",
                len(documents) - before,
                pdf.name,
            )

        except Exception:

            logger.exception(
                "Local PDF failed and was skipped: %s",
                pdf.name,
            )

    # --------------------------------------------------------
    # STRUCTURED QEC CONTACT DATA
    # --------------------------------------------------------

    documents.append(
        Document(
            page_content=QEC_CONTACT_DETAILS,
            metadata={
                "source": (
                    "JUW QEC Contact Us "
                    "(official details)"
                ),
                "document_name": "JUW QEC Contact Us",
                "document_type": "structured",
                "page": 0,
                "heading": "JUW QEC Contact Us",
            },
        )
    )

    # --------------------------------------------------------
    # OFFICIAL JUW WEBPAGES
    # --------------------------------------------------------

    for url in WEB_SOURCES:

        add_source_safely(
            documents,
            lambda u=url: [
                web_page_to_document(u)
            ],
            url,
        )

    # --------------------------------------------------------
    # OFFICIAL JUW POLICY PDFs
    # --------------------------------------------------------

    for url in QEC_POLICY_PDFS:

        add_source_safely(
            documents,
            lambda u=url: web_pdf_to_documents(u),
            url,
        )

    # --------------------------------------------------------
    # VALIDATE EXTRACTION
    # --------------------------------------------------------

    if not documents:

        logger.error(
            "No source documents were loaded."
        )

        sys.exit(1)

    logger.info(
        "Total source documents: %d",
        len(documents),
    )

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    logger.info(
        "Creating semantic chunks..."
    )

    chunks = split_documents(
        documents
    )

    logger.info(
        "Chunks before deduplication: %d",
        len(chunks),
    )

    # --------------------------------------------------------
    # DEDUPLICATION
    # --------------------------------------------------------

    chunks = deduplicate_chunks(
        chunks
    )

    logger.info(
        "Chunks after deduplication: %d",
        len(chunks),
    )

    if not chunks:

        logger.error(
            "No usable chunks were created."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    if lengths:

        average_length = (
            sum(lengths) / len(lengths)
        )

        logger.info(
            "Average chunk length: %.0f characters",
            average_length,
        )

        logger.info(
            "Shortest chunk: %d characters",
            min(lengths),
        )

        logger.info(
            "Longest chunk: %d characters",
            max(lengths),
        )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    logger.info(
        "Loading embedding model..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    logger.info(
        "Creating FAISS index..."
    )

    index = FAISS.from_documents(
        chunks,
        embeddings,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index.save_local(
        str(INDEX_DIR)
    )

    logger.info(
        "Index successfully written to: %s",
        INDEX_DIR,
    )

    logger.info("=" * 70)
    logger.info(
        "INDEX BUILD COMPLETE | %d chunks",
        len(chunks),
    )
    logger.info("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()