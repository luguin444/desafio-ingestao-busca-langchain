import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from providers_helper import EMBEDDING_PROVIDER, get_embedding_model

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def ingest_pdf():
    if not PDF_PATH:
        raise RuntimeError("PDF_PATH not defined")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not defined")
    if not COLLECTION_NAME:
        raise RuntimeError("PG_VECTOR_COLLECTION_NAME not defined")

    pdf_path = Path(PDF_PATH)
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).resolve().parent.parent / pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    print(f"[ingest] Loading PDF: {pdf_path}")
    docs = PyPDFLoader(str(pdf_path)).load()
    print(f"[ingest] Pages loaded: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Skip per-chunk character offset metadata — not needed for CLI-only RAG.
        add_start_index=False,
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Chunks generated: {len(chunks)}")

    print(f"[ingest] Embeddings provider: {EMBEDDING_PROVIDER}")
    embedding_model = get_embedding_model()

    store = PGVector(
        embeddings=embedding_model,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    print("[ingest] Inserting chunks into PGVector...")
    store.add_documents(chunks)
    print("[ingest] Ingestion completed.")


if __name__ == "__main__":
    ingest_pdf()