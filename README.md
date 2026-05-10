# MBA Software Engineering with AI Challenge — Full Cycle

Ingest a PDF into a vector database (PostgreSQL + pgVector) and run semantic search via CLI using LangChain. Supports OpenAI and Google (Gemini) as embedding/LLM providers.

## Prerequisites

- Python 3.13+
- Docker + Docker Compose
- An API key from OpenAI **or** Google AI Studio

## Project structure

```
.
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── empresa-faturamento-fundacao.pdf
└── src/
    ├── ingest.py             # Reads the PDF, generates embeddings, stores them in PGVector
    ├── search.py             # Search pipeline: embed question → top-k retrieval → LLM
    ├── chat.py               # Interactive CLI on top of search.py
    └── providers_helper.py   # Factory for embedding model and LLM (OpenAI or Google)
```

## 1. Configure environment variables

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

Relevant variables:

| Variable | Description | Example |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `GOOGLE_API_KEY` | Google AI Studio API key | `AIza...` |
| `EMBEDDING_PROVIDER` | `openai` or `google` | `openai` |
| `LLM_PROVIDER` | `openai` or `google` (defaults to `EMBEDDING_PROVIDER`) | `openai` |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embeddings model | `text-embedding-3-small` |
| `GOOGLE_EMBEDDING_MODEL` | Google embeddings model | `models/gemini-embedding-001` |
| `OPENAI_LLM_MODEL` | OpenAI chat model | `gpt-5-nano` |
| `GOOGLE_LLM_MODEL` | Google chat model | `gemini-2.5-flash-lite` |
| `DATABASE_URL` | SQLAlchemy/psycopg connection string for Postgres | `postgresql+psycopg://postgres:postgres@localhost:5433/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Logical collection name inside PGVector | `empresa-faturamento-fundacao-openai` |
| `PDF_PATH` | Path to the PDF to ingest (relative to project root) | `empresa-faturamento-fundacao.pdf` |

> You only need to fill the `API_KEY` for the provider you chose.

## 2. Start the database

```bash
docker compose up -d
```

The `docker-compose.yml` uses the `pgvector/pgvector:pg17` image and exposes port **5433** on the host (keeping 5432 free for any local Postgres you may have). The `vector` extension is created automatically on the first run of `ingest.py`.

Check the container status:

```bash
docker compose ps
```

## 3. Virtualenv + dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Ingest the PDF

```bash
python src/ingest.py
```

Expected output:

```
[ingest] Loading PDF: .../empresa-faturamento-fundacao.pdf
[ingest] Pages loaded: 34
[ingest] Chunks generated: 67
[ingest] Embeddings provider: openai
[ingest] Inserting chunks into PGVector...
[ingest] Ingestion completed.
```

The PDF is split into chunks of **1000 characters with 150 overlap**, embedded via the configured provider, and stored in PGVector.

## 5. Run the chat

```bash
python src/chat.py
```

Sample session:

```
Chat RAG — Pergunte sobre o PDF ingerido. Digite "sair" para encerrar.

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: R$ 10.000.000,00.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: sair
```

The search retrieves the **top 10 chunks** (k=10), assembles the context, and calls the LLM with a strict prompt: when the answer is not in the context, the model is instructed to reply `Não tenho informações necessárias para responder sua pergunta.`

## Switching providers (OpenAI ↔ Google)

Vector dimensions differ across providers (OpenAI = 1536, Google `gemini-embedding-001` truncated to 768). To avoid mixing dimensions in the same collection:

1. Clear the current collection or change `PG_VECTOR_COLLECTION_NAME` in `.env`.
2. Update `EMBEDDING_PROVIDER` and `LLM_PROVIDER`.
3. Run `python src/ingest.py` again.

Full reset (drops the Docker volume):

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```

## Database connection (debug)

| Field | Value |
|---|---|
| Host | `localhost` |
| Port | `5433` |
| Database | `rag` |
| User / Password | `postgres` / `postgres` |

Tables created by `langchain-postgres`:

- `langchain_pg_collection` — one row per collection.
- `langchain_pg_embedding` — one row per chunk (id, document, embedding, cmetadata).
