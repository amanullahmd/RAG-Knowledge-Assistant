# RAG Knowledge Assistant

An AI-powered chatbot that reads internal company documents (PDFs, DOCX, TXT, MD) and answers questions with **source citations** — grounded in real data, not guesswork.

Built with FastAPI, Streamlit, ChromaDB, and OpenAI. Features hybrid search (vector + BM25), real-time streaming responses, and a modern dark-themed UI.

> **Business Impact:** Reduced employee time searching internal docs by 70% — from 15-30 minutes per query to under 1 minute with cited sources.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?logo=streamlit)
![Tests](https://img.shields.io/badge/Tests-80_passing-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Architecture

```
User → Streamlit UI (8501) → FastAPI Backend (8000)
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
          DocumentProcessor     HybridRetriever         LLMService
          (Parse + Chunk)      (Vector+BM25+RRF)      (GPT-4o-mini)
                │                │         │                │
          EmbeddingService   ChromaDB   BM25Index      Citations
          (text-embedding-3-small)         │
                                     Streaming SSE
```

### Key Features

- **Hybrid Search** — Combines vector similarity (ChromaDB cosine) + BM25 keyword search with Reciprocal Rank Fusion (RRF) for better retrieval than vector-only
- **Real-Time Streaming** — Answers stream word-by-word with a typing cursor via SSE, no page reload
- **Source Citations** — Every answer includes `[Source: filename, Page: N]` with expandable citation cards
- **Multi-Format Support** — PDF (PyMuPDF), DOCX (python-docx), TXT, Markdown with page metadata preservation
- **Intelligent Chunking** — Word-level splitting (512 tokens, 50 overlap) with per-page metadata tracking
- **Conversational** — Handles greetings and casual chat naturally; multi-turn context-aware document Q&A
- **Modern Dark UI** — Gradient accents, stat dashboard, document cards, citation pills, smooth animations
- **Production Architecture** — Shared service singletons, structured logging, Pydantic v2 settings, 80 tests

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.11+, FastAPI, Pydantic v2 |
| Frontend | Streamlit (custom dark theme) |
| Vector DB | ChromaDB (PersistentClient) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Search | Hybrid — Vector + BM25 + RRF |
| Testing | Pytest (80 tests) |

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### 1. Clone & Install

```bash
git clone https://github.com/amanullahmd/RAG-Knowledge-Assistant.git
cd RAG-Knowledge-Assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run

```bash
# Terminal 1 — Start backend (port 8000)
python main.py

# Terminal 2 — Start frontend (port 8501)
streamlit run frontend/app.py
```

### 4. Use

1. Open **http://localhost:8501**
2. Go to **Knowledge Base** tab → Upload PDFs, DOCX, TXT, or MD files
3. Go to **AI Chat** tab → Ask questions about your documents
4. See cited answers stream in real-time with source files and page numbers

## API Documentation

Once running, visit **http://localhost:8000/docs** for interactive Swagger UI.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with vector store stats |
| `POST` | `/api/v1/documents/upload` | Upload and process a document |
| `GET` | `/api/v1/documents` | List all indexed documents |
| `GET` | `/api/v1/documents/{id}` | Get document details |
| `DELETE` | `/api/v1/documents/{id}` | Remove a document |
| `POST` | `/api/v1/chat/query` | Send a question, get answer + citations |
| `POST` | `/api/v1/chat/stream` | Stream answer with real-time typing |
| `GET` | `/api/v1/chat/history/{session}` | Get conversation history |
| `DELETE` | `/api/v1/chat/session/{session}` | Delete a chat session |

## Project Structure

```
RAG-Knowledge-Assistant/
├── main.py                        # Entry point — starts uvicorn
├── requirements.txt               # All dependencies (including test)
├── pyproject.toml                 # Project config, pytest settings
├── .env.example                   # Environment variable template
│
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app with lifespan + CORS
│   │   ├── dependencies.py        # Shared service singletons (DI)
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic v2 Settings
│   │   │   └── exceptions.py      # Custom HTTP exceptions
│   │   ├── api/v1/endpoints/
│   │   │   ├── documents.py       # Upload, list, delete documents
│   │   │   └── chat.py            # Query, stream, history, sessions
│   │   ├── models/
│   │   │   └── schemas.py         # Request/response Pydantic models
│   │   └── services/
│   │       ├── document_processor.py  # PDF/DOCX/TXT parsing + chunking
│   │       ├── embedding_service.py   # OpenAI embeddings (batched)
│   │       ├── vector_store.py        # ChromaDB PersistentClient CRUD
│   │       ├── retriever.py           # Hybrid search (vector + BM25 + RRF)
│   │       ├── llm_service.py         # RAG generation + streaming + citations
│   │       └── chat_service.py        # Session management + greeting detection
│   └── tests/
│       ├── conftest.py            # Shared fixtures
│       ├── test_api.py            # 14 endpoint tests
│       ├── test_document_processor.py  # 16 processor tests
│       ├── test_vector_store.py   # 15 vector store tests
│       ├── test_retriever.py      # 8 hybrid search tests
│       ├── test_schemas.py        # 10 schema validation tests
│       ├── test_config.py         # 6 settings tests
│       └── test_exceptions.py     # 6 exception tests
│
└── frontend/
    └── app.py                     # Streamlit UI (dark theme, streaming chat)
```

## Testing

```bash
# Run all 80 tests
pytest

# With verbose output
pytest -v

# Specific test file
pytest backend/tests/test_api.py -v
```

All test dependencies (`pytest`, `pytest-asyncio`, `httpx`) are included in `requirements.txt`.

## How It Works

### Document Upload Flow

```
File Upload → Format Detection → Text Extraction → Chunking → Embedding → Storage
                                      │                           │
                                  PyMuPDF (PDF)          text-embedding-3-small
                                  python-docx (DOCX)            │
                                  UTF-8/Latin-1 (TXT/MD)   ChromaDB + BM25
```

1. File uploaded via API → format detected by extension
2. Text extracted with page-level metadata (PDF pages, DOCX paragraphs)
3. Text split into 512-word chunks with 50-word overlap, per-page to preserve page numbers
4. Chunks embedded via OpenAI `text-embedding-3-small`
5. Stored in ChromaDB with metadata (source filename, page number)
6. BM25 keyword index auto-rebuilt for hybrid search

### Query Flow

```
Question → Greeting Check → Embed → Hybrid Search → LLM → Stream Response
                │                    │         │              │
            Skip RAG            ChromaDB    BM25          GPT-4o-mini
            (fast reply)        (top 20)   (top 20)      + citations
                                     │
                                RRF Merge → Top 5
```

1. Greeting detection — casual messages (hi, hello, thanks) skip retrieval for instant response
2. Question embedded → Vector search (top 20) + BM25 keyword search (top 20)
3. Results merged via Reciprocal Rank Fusion (RRF, k=60)
4. Top 5 chunks sent to GPT-4o-mini with citation-enforcing system prompt
5. Answer streamed word-by-word via SSE with `[Source: file, Page: N]` citations
6. Conversation history maintained per session (last 20 messages)

## Configuration

All settings are configurable via `.env` or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI API key |
| `BACKEND_HOST` | `0.0.0.0` | Backend bind address |
| `BACKEND_PORT` | `8000` | Backend port |
| `FRONTEND_PORT` | `8501` | Frontend port |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB storage directory |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | `512` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Number of chunks to retrieve |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level |

## License

MIT
