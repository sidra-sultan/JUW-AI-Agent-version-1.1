# JUW AI Agent - Remediation Log

## Critical security issue

### Exposed API key
**Problem:** The supplied working tree contained a real-looking Groq API key in `.env`.

**Action:** The distributable copy removes `.env` entirely. `.env.example` remains the only configuration template.

**Required user action:** Revoke/rotate the exposed key. Removing it from this ZIP does not invalidate a key that was already exposed.

---

## Retrieval

### Before
The API used `similarity_search(..., k=3)` and sent all three passages directly to the LLM.

### After
Retrieval now uses:

```text
Query
  ↓
Normalization + small abbreviation expansion
  ↓
Semantic candidates (default 12)
  +
Lexical candidates (default 12)
  ↓
Score fusion
  ↓
Duplicate removal
  ↓
Minimum relevance filtering
  ↓
Source/page diversity selection
  ↓
Top 5 context chunks
  ↓
Grounded LLM
```

This improves recall without blindly dumping a large number of chunks into the context.

A lightweight lexical fallback was chosen instead of adding Elasticsearch/BM25 infrastructure. It uses the existing FAISS documents and a TF-IDF-like token-overlap score.

A full MMR/reranker model was deliberately **not** added. For this project, that would add complexity and model/runtime cost before the simpler hybrid approach has been measured.

---

## Chunking and metadata

### Before
The indexer used a generic fixed-size recursive splitter and retained only limited loader metadata.

### After
Chunking still uses `RecursiveCharacterTextSplitter`, but now:

- processes each PDF page as its own logical unit;
- uses paragraph/newline/sentence-aware separators;
- uses a smaller overlap;
- attempts to identify headings;
- records `source`, `document_name`, `document_type`, `page`, `url` where applicable;
- creates a stable `chunk_id`.

This is an incremental improvement rather than an unnecessary full document-parser rewrite.

---

## PDF ingestion

### Before
`PyPDFLoader` was called over every PDF in one loop. A failure could abort the rebuild.

### After
PDFs are processed independently with `pypdf`.

Failures are isolated at file/page level and logged. Encrypted PDFs are detected. If an encrypted PDF is still readable, extraction is attempted.

Pages with little/no text are reported as candidates for OCR. OCR was not added blindly because the current local corpus contains extractable text on its supplied PDFs.

The rebuild refuses to overwrite the existing index when zero usable chunks are produced.

---

## Web ingestion

### Before
Remote sources had a single request path and a long hard-coded timeout.

### After
Remote fetching has:

- connect/read timeouts;
- bounded retries;
- increasing retry delay;
- HTTP status validation;
- redirect support;
- content-type validation;
- per-source failure isolation.

One unavailable official page no longer prevents the remaining knowledge base from being rebuilt.

---

## Duplicate and context handling

Retrieved chunks are deduplicated by content hash.

The final selection also applies a small source/page diversity penalty so several near-identical passages from the same source are less likely to consume the entire context window.

---

## Grounding

The system prompt now explicitly requires:

- answering only from supplied JUW evidence;
- refusing unsupported university facts;
- refusing questions about other universities;
- never inventing dates, fees, departments, people, URLs, contacts, or page numbers;
- using source/page metadata when available.

The LLM still makes the final semantic decision, so retrieval evaluation remains necessary. Prompt wording alone is not treated as a hallucination guarantee.

---

## Conversation/session handling

### Before
The browser history could grow without bound and eventually exceed the backend's validation limit.

### After
The frontend keeps a bounded history and generates a `session_id`.

The backend accepts a bounded history and sends only the most recent configured number of exchanges to the LLM.

This remains intentionally stateless on the server. A session identifier is available for request correlation without introducing Redis/database infrastructure.

---

## API and error handling

The backend now separates:

- retrieval failures;
- missing configuration;
- rate limiting;
- AI timeout;
- AI connection failure;
- upstream API errors;
- unexpected internal failures.

Users receive safe, human-readable messages. Raw SDK exceptions are not returned to the browser.

CORS origins are configurable rather than hard-coded.

---

## Frontend

Existing JUW branding and layout were preserved.

Existing fixes remain:

- bounded history;
- readable error parsing;
- typing indicator;
- disabled send button during requests;
- source display;
- clear-chat behavior.

The frontend and backend request contract now also supports `session_id`.

---

## Testing performed in this remediation

### Passed
- Python syntax compilation for `backend/app.py`, `scripts/build_index.py`, and `tests/evaluate.py`.
- JavaScript syntax check with Node.
- Local PDF extraction inspection across all six supplied PDFs.
- Secret scan of the distributable working tree after removing `.env`.
- Git history check confirmed `.env` was not committed in the supplied repository history.

### PDF inspection result
All six supplied local PDFs could be opened with `pypdf`. One PDF is encrypted/protected but remained extractable. Several pages have little/no extractable text, so they are explicitly treated as OCR candidates rather than falsely claiming OCR support.

### Not run
A full application/integration test could not be executed in this audit environment because the uploaded project dependencies (`groq`, `langchain-community`, `langchain-huggingface`, `sentence-transformers`, and FAISS) are not installed and this environment has no package-index/network access for installing them.

The live 16-question LLM evaluation therefore has **not** been claimed as passed.

The remote JUW web/PDF rebuild was also not executed here because it requires the project dependencies and external network access.

---

## Remaining issues

1. OCR remains optional and must be added if scanned JUW pages are important to retrieval.
2. The current bundled FAISS index was not rebuilt in this environment.
3. A real retrieval/answer benchmark should be run after dependencies are installed and the index is rebuilt.
4. The current system is intentionally stateless. If persistent multi-user conversations become a requirement, a small database can be considered later.
5. A learned reranker/MMR implementation should only be added after measuring whether the hybrid retrieval approach still misses important questions.
