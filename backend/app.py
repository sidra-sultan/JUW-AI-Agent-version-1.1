"""JUW RAG web API.

Run with:
    uvicorn backend.app:app --reload
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Groq,
    RateLimitError,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("juw-agent")

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "faiss_index"
FRONTEND_DIR = ROOT / "frontend"

load_dotenv(ROOT / ".env")


# ============================================================
# GROQ CONFIGURATION
# ============================================================

MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)

GROQ_API_KEY_PRIMARY = os.getenv(
    "GROQ_API_KEY_PRIMARY"
)

GROQ_API_KEY_FALLBACK = os.getenv(
    "GROQ_API_KEY_FALLBACK"
)


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

TOP_K_CANDIDATES = int(
    os.getenv(
        "TOP_K_CANDIDATES",
        "12",
    )
)

FINAL_CONTEXT_CHUNKS = int(
    os.getenv(
        "FINAL_CONTEXT_CHUNKS",
        "5",
    )
)

MAX_CHUNK_CHARS_FOR_LLM = int(
    os.getenv(
        "MAX_CHUNK_CHARS_FOR_LLM",
        "1200",
    )
)

RETRIEVAL_MIN_SCORE = float(
    os.getenv(
        "RETRIEVAL_MIN_SCORE",
        "0.20",
    )
)


# ============================================================
# CONVERSATION CONFIGURATION
# ============================================================

HISTORY_TURNS_SENT_TO_MODEL = int(
    os.getenv(
        "HISTORY_TURNS_SENT_TO_MODEL",
        "2",
    )
)

MAX_HISTORY_ITEMS = int(
    os.getenv(
        "MAX_HISTORY_ITEMS",
        "40",
    )
)


# ============================================================
# RESPONSE MESSAGES
# ============================================================

OUT_OF_SCOPE = (
    "I am configured to answer queries regarding "
    "Jinnah University for Women (JUW) only."
)

NO_ANSWER = (
    "Information regarding this specific query is currently "
    "not available in my official JUW database."
)

BUSY_MESSAGE = (
    "The AI service is temporarily unavailable. "
    "Please try again shortly."
)

# ============================================================
# LLM SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are the official information assistant for Jinnah University for Women (JUW).

Answer ONLY from the supplied JUW sources.
Do not use general knowledge and do not invent facts.

If the supplied sources do not answer the question, reply exactly:
{NO_ANSWER}

If the question is about another university, reply exactly:
{OUT_OF_SCOPE}

Reply in English or Roman Urdu only.
Never use Urdu or Arabic script.

ANSWERING STYLE:
- Give complete, clear, helpful, and natural answers.
- Answer the user's exact question first.
- Use as much detail as necessary, but do not add irrelevant information.
- Rewrite information naturally in your own words.
- Do not copy large portions of the source text.
- Do not repeat the same information.
- For simple questions, give a concise direct answer.
- For broad questions, organize the answer into useful sections.
- For procedures, explain the steps in the correct order.
- Keep related information together.

ACCURACY RULES:
- Answer only what is supported by the supplied JUW sources.
- Never guess or fill missing information using general knowledge.
- Never invent a fee, date, deadline, policy, department, program,
  faculty member, designation, facility, contact detail, eligibility
  requirement, admission procedure, or other fact.
- If the supplied sources contain insufficient information, reply exactly:
{NO_ANSWER}

DEPARTMENT AND FACULTY ACCURACY:
- Do not mix information from different departments.
- Do not assume that a person belongs to a department just because
  their name appears somewhere in the retrieved context.
- Only associate a faculty member with a department when the supplied
  source explicitly establishes that relationship.
- Do not list faculty from Chemistry, Biology, Biochemistry,
  Microbiology, Biotechnology, Zoology, or other departments when the
  user asks specifically about Computer Science.
- Do not claim that a faculty list is complete unless the supplied
  source clearly provides a complete list.
- If the requested department-specific information is not reliably
  available, reply exactly:
{NO_ANSWER}

RESPONSE FORMAT:
- Use clean plain text only.
- Do NOT use Markdown.
- Do NOT use asterisks (*) for bold or emphasis.
- Do NOT use Markdown tables.
- Do NOT use Markdown headings such as # or ##.
- Do NOT use HTML tags such as <br>, <p>, <b>, <strong>, <ul>, or <li>.
- Do NOT use backticks for formatting.
- Use simple section titles without special formatting.
- Use numbered lists for step-by-step procedures.
- Use the bullet character "•" for lists.
- Leave a blank line between major sections.
- Keep the response visually organized and easy to read.

SOURCE PRIVACY RULES:
- Use the supplied sources only as internal evidence.
- NEVER mention source filenames in your answer.
- NEVER mention page numbers in your answer.
- NEVER mention source URLs in your answer.
- NEVER expose metadata in your answer.
- NEVER expose retrieval scores.
- NEVER write "Sources:" followed by documents.
- NEVER use citation formats such as [Source: ...].
- NEVER reproduce internal source labels.
- Do not reveal how the information was retrieved.

The user should receive a clean, natural, well-organized answer,
not an explanation of the retrieval system.
"""

# ============================================================
# QUERY EXPANSION
# ============================================================

QUERY_EXPANSIONS = {
    "bscs": "BS Computer Science",
    "bs cs": "BS Computer Science",
    "cs": "Computer Science",
    "bba": "Bachelor of Business Administration",
    "mba": "Master of Business Administration",
    "ms": "MS",
    "mphil": "M.Phil",
    "phd": "PhD",
    "admission": "admissions",
    "fee": "fees fee structure",
}


_TOKEN_RE = re.compile(
    r"[a-z0-9]+",
    re.I,
)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "does",
    "do",
    "tell",
    "about",
    "can",
    "please",
    "there",
    "this",
    "that",
    "have",
    "has",
    "had",
    "you",
    "your",
}


def normalize_query(query: str) -> str:
    """Normalize whitespace and expand university-specific abbreviations."""

    normalized = " ".join(
        query.lower().split()
    )

    for term, expansion in QUERY_EXPANSIONS.items():

        if term in normalized:

            normalized = (
                f"{normalized} "
                f"{expansion.lower()}"
            )

    return normalized


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(
            text.lower()
        )
        if token not in _STOPWORDS
        and len(token) > 1
    ]


# ============================================================
# DOCUMENT HELPERS
# ============================================================

def _content_key(document) -> str:
    return hashlib.sha1(
        document.page_content.encode(
            "utf-8",
            "ignore",
        )
    ).hexdigest()


def source_label(document) -> str:
    """
    Generate an internal source label.

    This is used internally by the backend.
    It should NOT be exposed directly to the user.
    """

    metadata = document.metadata or {}

    source = (
        metadata.get("document_name")
        or metadata.get("source")
        or "JUW knowledge base"
    )

    page = metadata.get("page")

    if isinstance(page, int):

        return (
            f"{Path(str(source)).name}, "
            f"page {page + 1}"
        )

    return Path(
        str(source)
    ).name


# ============================================================
# LEXICAL SEARCH
# ============================================================

def _build_lexical_index(vectorstore):
    """Build a lightweight in-memory lexical index."""

    entries = []

    document_frequency = Counter()

    ids = getattr(
        vectorstore,
        "index_to_docstore_id",
        {},
    )

    docstore = getattr(
        vectorstore,
        "docstore",
        None,
    )

    for doc_id in ids.values():

        document = (
            docstore.search(doc_id)
            if docstore
            else None
        )

        if (
            document is None
            or not getattr(
                document,
                "page_content",
                "",
            ).strip()
        ):
            continue

        tokens = tokenize(
            document.page_content
        )

        counts = Counter(tokens)

        entries.append(
            (
                document,
                counts,
            )
        )

        document_frequency.update(
            counts.keys()
        )

    total = max(
        len(entries),
        1,
    )

    idf = {
        token: math.log(
            (1 + total)
            / (1 + frequency)
        ) + 1.0
        for token, frequency
        in document_frequency.items()
    }

    logger.info(
        "Lexical fallback indexed %d chunks",
        len(entries),
    )

    return entries, idf


# ============================================================
# VECTOR STORE
# ============================================================

embeddings = None
vectorstore = None
lexical_entries = None
lexical_idf = None


# ============================================================
# GROQ CLIENTS
# ============================================================

groq_clients = []

if GROQ_API_KEY_PRIMARY:

    groq_clients.append(
        (
            "primary",
            Groq(
                api_key=GROQ_API_KEY_PRIMARY
            ),
        )
    )


if GROQ_API_KEY_FALLBACK:

    groq_clients.append(
        (
            "fallback",
            Groq(
                api_key=GROQ_API_KEY_FALLBACK
            ),
        )
    )


logger.info(
    "Configured %d Groq API client(s)",
    len(groq_clients),
)


def get_vectorstore():
    global embeddings
    global vectorstore
    global lexical_entries
    global lexical_idf

    if vectorstore is None:

        if not INDEX_DIR.exists():

            raise RuntimeError(
                f"FAISS index missing: "
                f"{INDEX_DIR}"
            )

        logger.info(
            "Loading embedding model and FAISS index"
        )

        embeddings = HuggingFaceEmbeddings(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            )
        )

        vectorstore = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        (
            lexical_entries,
            lexical_idf,
        ) = _build_lexical_index(
            vectorstore
        )

        logger.info(
            "FAISS index loaded"
        )

    return vectorstore


# ============================================================
# SEARCH
# ============================================================

def _lexical_search(
    query: str,
    limit: int,
) -> list[tuple[object, float]]:

    if (
        not lexical_entries
        or not lexical_idf
    ):
        return []

    query_tokens = tokenize(
        query
    )

    if not query_tokens:
        return []

    query_weights = {
        token: lexical_idf.get(
            token,
            1.0,
        )
        for token in query_tokens
    }

    query_mass = (
        sum(
            query_weights.values()
        )
        or 1.0
    )

    scored = []

    for document, counts in lexical_entries:

        matched = sum(
            weight
            for token, weight
            in query_weights.items()
            if counts.get(token)
        )

        if matched:

            score = (
                matched
                / query_mass
            )

            scored.append(
                (
                    document,
                    min(
                        score,
                        1.0,
                    ),
                )
            )

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored[:limit]


def _semantic_search(
    query: str,
    limit: int,
) -> list[tuple[object, float]]:

    store = get_vectorstore()

    try:

        results = (
            store.similarity_search_with_relevance_scores(
                query,
                k=limit,
            )
        )

        return [
            (
                doc,
                max(
                    0.0,
                    min(
                        float(score),
                        1.0,
                    ),
                ),
            )
            for doc, score in results
        ]

    except Exception:

        logger.warning(
            "Calibrated FAISS scores unavailable; "
            "using distance scores"
        )

        results = (
            store.similarity_search_with_score(
                query,
                k=limit,
            )
        )

        return [
            (
                doc,
                1.0
                / (
                    1.0
                    + max(
                        float(distance),
                        0.0,
                    )
                ),
            )
            for doc, distance in results
        ]


def _dedupe_candidates(
    candidates,
):
    by_content = {}

    for document, score in candidates:

        key = _content_key(
            document
        )

        if (
            key not in by_content
            or score
            > by_content[key][1]
        ):

            by_content[key] = (
                document,
                score,
            )

    return list(
        by_content.values()
    )


def _select_diverse(
    candidates,
    limit: int,
):
    """Select high-scoring chunks while reducing repetition."""

    selected = []

    seen_sources = Counter()

    remaining = list(
        candidates
    )

    while (
        remaining
        and len(selected) < limit
    ):

        best_index = 0

        best_value = float(
            "-inf"
        )

        selected_pages = {
            (
                (
                    d.metadata
                    or {}
                ).get(
                    "document_name"
                )
                or (
                    d.metadata
                    or {}
                ).get(
                    "source"
                )
                or "unknown",
                (
                    d.metadata
                    or {}
                ).get(
                    "page"
                ),
            )
            for d, _ in selected
        }

        for index, (
            document,
            score,
        ) in enumerate(
            remaining
        ):

            metadata = (
                document.metadata
                or {}
            )

            source = (
                metadata.get(
                    "document_name"
                )
                or metadata.get(
                    "source"
                )
                or "unknown"
            )

            page = metadata.get(
                "page"
            )

            key = (
                str(source),
                page,
            )

            penalty = (
                0.06
                * seen_sources[
                    str(source)
                ]
                + (
                    0.03
                    if key
                    in selected_pages
                    else 0.0
                )
            )

            value = (
                score
                - penalty
            )

            if value > best_value:

                best_value = value
                best_index = index

        document, score = (
            remaining.pop(
                best_index
            )
        )

        selected.append(
            (
                document,
                score,
            )
        )

        source = (
            (
                document.metadata
                or {}
            ).get(
                "document_name"
            )
            or (
                document.metadata
                or {}
            ).get(
                "source"
            )
            or "unknown"
        )

        seen_sources[
            str(source)
        ] += 1

    return selected


def retrieve_documents(
    query: str,
):
    """Hybrid semantic + lexical retrieval."""

    normalized = normalize_query(
        query
    )

    semantic = _semantic_search(
        normalized,
        TOP_K_CANDIDATES,
    )

    lexical = _lexical_search(
        normalized,
        TOP_K_CANDIDATES,
    )

    semantic_map = {
        _content_key(doc): score
        for doc, score in semantic
    }

    lexical_map = {
        _content_key(doc): score
        for doc, score in lexical
    }

    documents = {
        _content_key(doc): doc
        for doc, _ in (
            semantic + lexical
        )
    }

    combined = []

    for key, document in documents.items():

        semantic_score = (
            semantic_map.get(
                key,
                0.0,
            )
        )

        lexical_score = (
            lexical_map.get(
                key,
                0.0,
            )
        )

        if (
            semantic_score
            and lexical_score
        ):

            score = (
                0.70
                * semantic_score
                + 0.30
                * lexical_score
            )

        elif semantic_score:

            score = semantic_score

        else:

            score = (
                0.75
                * lexical_score
            )

        combined.append(
            (
                document,
                score,
            )
        )

    combined.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    relevant = [
        item
        for item in combined
        if item[1]
        >= RETRIEVAL_MIN_SCORE
    ]

    if not relevant:

        logger.info(
            "No candidate passed retrieval threshold"
        )

        return []

    selected = _select_diverse(
        relevant,
        FINAL_CONTEXT_CHUNKS,
    )

    logger.info(
        "Query retrieved %d semantic + "
        "%d lexical candidates; selected %d",
        len(semantic),
        len(lexical),
        len(selected),
    )

    return selected


# ============================================================
# API MODELS
# ============================================================

class ChatMessage(BaseModel):

    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        min_length=1,
        max_length=4000,
    )


class ChatRequest(BaseModel):

    session_id: str | None = Field(
        default=None,
        min_length=8,
        max_length=100,
    )

    message: str = Field(
        min_length=2,
        max_length=1000,
    )

    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_ITEMS,
    )


class ChatResponse(BaseModel):

    answer: str
    sources: list[str]


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="JUW AI Agent",
    version="2.0.0",
)


_origins_env = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,"
    "http://127.0.0.1:8000",
)


ALLOWED_ORIGINS = [
    origin.strip()
    for origin in _origins_env.split(",")
    if origin.strip()
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
    ],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "index_available": INDEX_DIR.exists(),
        "model": MODEL,
        "groq_clients": len(
            groq_clients
        ),
        "retrieval": {
            "candidate_k": TOP_K_CANDIDATES,
            "final_chunks": FINAL_CONTEXT_CHUNKS,
            "max_chunk_chars": (
                MAX_CHUNK_CHARS_FOR_LLM
            ),
        },
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    if not groq_clients:

        raise HTTPException(
            500,
            "No Groq API key is configured. "
            "Add at least one API key to "
            "the environment.",
        )

    logger.info(
        "Query received | session=%s | text=%s",
        request.session_id
        or "anonymous",
        request.message[:120],
    )


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    try:

        retrieved = retrieve_documents(
            request.message
        )

    except Exception as error:

        logger.exception(
            "Retrieval failed"
        )

        raise HTTPException(
            500,
            "Could not search the JUW "
            "knowledge base right now.",
        ) from error


    if not retrieved:

        logger.info(
            "No relevant JUW documents found"
        )

        return ChatResponse(
            answer=NO_ANSWER,
            sources=[],
        )


    # --------------------------------------------------------
    # BUILD INTERNAL CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for document, score in retrieved:

        label = source_label(
            document
        )

        content = (
            document.page_content
            .strip()
            [:MAX_CHUNK_CHARS_FOR_LLM]
        )

        context_parts.append(
            f"[INTERNAL SOURCE: {label}]\n"
            f"{content}"
        )


    context = (
        "\n\n--- INTERNAL SOURCE ---\n"
        .join(context_parts)
    )


    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

    recent_history = request.history[
        -(HISTORY_TURNS_SENT_TO_MODEL * 2):
    ]


    safe_history = [
        item.model_dump()
        for item in recent_history
    ]


    # --------------------------------------------------------
    # LLM MESSAGES
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                f"JUW INTERNAL SOURCES:\n"
                f"{context}"
            ),
        },
        *safe_history,
        {
            "role": "user",
            "content": request.message,
        },
    ]


    # --------------------------------------------------------
    # GROQ REQUEST WITH FALLBACK
    # --------------------------------------------------------

    completion = None
    last_error = None

    for key_name, client in groq_clients:

        logger.info(
            "Trying Groq %s key",
            key_name,
        )

        try:

            completion = (
                client
                .chat
                .completions
                .create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000,
                    timeout=20,
                )
            )

            logger.info(
                "Groq %s key succeeded",
                key_name,
            )

            break


        except RateLimitError as error:

            last_error = error

            logger.warning(
                "Groq %s key rate-limited: %s",
                key_name,
                error,
            )

            # Try the next key immediately.
            continue


        except APITimeoutError as error:

            last_error = error

            logger.warning(
                "Groq %s key timed out",
                key_name,
            )

            # Try fallback.
            continue


        except APIConnectionError as error:

            last_error = error

            logger.warning(
                "Groq %s key connection failed",
                key_name,
            )

            # Try fallback.
            continue


        except APIStatusError as error:

            last_error = error

            status_code = getattr(
                error,
                "status_code",
                None,
            )

            logger.warning(
                "Groq %s key returned status %s: %s",
                key_name,
                status_code,
                error,
            )

            # Retry with another key only for
            # authentication, rate-limit, or
            # temporary server errors.
            if status_code in {
                401,
                429,
                500,
                502,
                503,
                504,
            }:

                continue


            # Permanent request/model errors
            # should not be hidden by key rotation.
            raise HTTPException(
                502,
                "The AI service rejected the request. "
                "Please try again.",
            ) from error


        except Exception as error:

            last_error = error

            logger.exception(
                "Unexpected error with Groq %s key",
                key_name,
            )

            # Try the next key.
            continue


    # --------------------------------------------------------
    # ALL GROQ CLIENTS FAILED
    # --------------------------------------------------------

    if completion is None:

        logger.error(
            "All configured Groq API keys failed"
        )

        raise HTTPException(
            503,
            BUSY_MESSAGE,
        ) from last_error


    # --------------------------------------------------------
    # CLEAN RESPONSE
    # --------------------------------------------------------

    reply = (
        completion
        .choices[0]
        .message
        .content
        or ""
    ).strip()


    if not reply:

        reply = NO_ANSWER


    # --------------------------------------------------------
    # INTERNAL SOURCE INFORMATION
    # --------------------------------------------------------

    sources = list(
        dict.fromkeys(
            source_label(document)
            for document, _ in retrieved
        )
    )


    logger.info(
        "Response generated from "
        "%d selected chunks",
        len(retrieved),
    )


    return ChatResponse(
        answer=reply,
        sources=sources,
    )


# ============================================================
# FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory=str(FRONTEND_DIR),
        html=True,
    ),
    name="frontend",
)