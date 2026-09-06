# JUW AI Agent

A FastAPI + FAISS + Hugging Face embeddings + Groq Retrieval-Augmented Generation (RAG) assistant for **Jinnah University for Women (JUW)**.

The application serves the frontend and API from one FastAPI process and answers from the bundled JUW knowledge base.

## Architecture

```text
Browser
  │
  ├── bounded chat history + session_id
  │
  ▼
FastAPI /api/chat
  │
  ├── query normalization / small abbreviation expansion
  │
  ├── semantic retrieval (top candidate set)
  │
  ├── lightweight lexical fallback
  │
  ├── score fusion + duplicate removal
  │
  ├── source/page diversity selection
  │
  └── bounded context
  │
  ▼
Grounded Groq LLM
  │
  ▼
Answer + source labels
```

The goal is not "more chunks". The goal is the **smallest set of relevant, diverse evidence** needed to answer correctly.

## Project layout

```text
juw-ai-agent/
├── backend/app.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── assets/
├── data/data/              # JUW PDFs
├── data/faiss_index/       # persisted FAISS index
├── scripts/build_index.py  # resilient ingestion/index rebuild
├── tests/evaluate.py       # 16-case live evaluation harness
├── requirements.txt
├── .env.example
└── CHANGES.md
```

## Run locally

1. Install Python 3.10+.
2. Create and activate a virtual environment:

   **Windows PowerShell**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your **new** Groq API key:
   ```text
   GROQ_API_KEY=...
   GROQ_MODEL=openai/gpt-oss-20b
   ```

5. Start the application:
   ```powershell
   uvicorn backend.app:app --reload
   ```

6. Open `http://127.0.0.1:8000`.

The embedding model is loaded lazily when the first retrieval request arrives.

## Configuration

Important values can be changed through environment variables:

```text
GROQ_API_KEY
GROQ_MODEL
ALLOWED_ORIGINS
TOP_K_CANDIDATES=12
FINAL_CONTEXT_CHUNKS=5
MAX_CHUNK_CHARS_FOR_LLM=1800
RETRIEVAL_MIN_SCORE=0.20
HISTORY_TURNS_SENT_TO_MODEL=3
MAX_HISTORY_ITEMS=40
REQUEST_TIMEOUT=30
MAX_RETRIES=3
RETRY_DELAY=1.5
LOG_LEVEL=INFO
```

The defaults are intentionally modest for a university project.

## Rebuild the knowledge index

Put official/current JUW PDFs in `data/data/`, then run:

```powershell
python scripts/build_index.py
```

The ingestion pipeline:

- processes each PDF independently;
- detects encrypted/protected PDFs and attempts readable extraction;
- isolates page-level extraction failures;
- logs pages with little/no extractable text instead of pretending OCR happened;
- retries remote requests with bounded timeouts;
- validates HTTP responses/content types;
- continues when one source fails;
- extracts selected official JUW webpages;
- adds selected QEC policy PDFs;
- preserves page/source/document metadata;
- chunks within document/page boundaries;
- creates stable chunk IDs;
- refuses to overwrite the index if no usable chunks were produced.

OCR is **not** silently enabled. Scanned PDFs with insufficient text are reported so OCR can be added deliberately if the corpus proves to need it.

## Evaluation

Run:

```powershell
python tests/evaluate.py
```

The release harness contains 16 representative checks covering:

- JUW scope;
- BS eligibility;
- programme-specific marks;
- admission-test weighting;
- contact information;
- programmes;
- faculties/departments;
- admissions;
- credit-hour terminology;
- academic-calendar information;
- out-of-scope questions;
- unavailable/fabricated information.

The script reports the actual result and retrieved sources. It does **not** claim an accuracy percentage without running the live cases.

## Security

- `.env` is ignored and is not part of the distributable repository.
- Never put API keys in source code, commits, screenshots, prompts, or issue reports.
- If an API key has ever been exposed, revoke/rotate it immediately and create a replacement key.
- Logs deliberately avoid API keys and raw SDK exception bodies.
- Only the local project-generated FAISS index is loaded with FAISS's required deserialization flag.

## GitHub

Before pushing, verify:

```powershell
git status
git check-ignore .env
git diff --check
```

Do not commit `.env`.

Also verify that the bundled PDFs and index are legally permitted to be redistributed.

## Deployment

This project can run as one persistent FastAPI service because the API also serves the frontend.

Example start command:

```text
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

Use a host that can keep a Python process running with enough memory for the embedding model. Configure the API key and other environment variables in the host dashboard rather than committing them.

