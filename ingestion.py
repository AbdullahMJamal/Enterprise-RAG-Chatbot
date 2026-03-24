"""
ingestion.py
────────────
Handles all document ingestion logic:
  1. Load a PDF from disk with PyPDFLoader
  2. Chunk the text with RecursiveCharacterTextSplitter
  3. Embed chunks via OpenAI text-embedding-3-small
  4. Persist / update a local FAISS vector store at db/faiss_db/

This module is intentionally side-effect free at import time so that
app.py can safely import it without triggering any heavy work.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── LangChain Imports ─────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Load environment variables from a .env file (OPENAI_API_KEY, etc.) ────────
load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
FAISS_DB_PATH   = "db/faiss_db"         # Directory where FAISS index is saved
CHUNK_SIZE      = 1000                  # Max characters per chunk
CHUNK_OVERLAP   = 100                   # Overlap between consecutive chunks
EMBEDDING_MODEL = "text-embedding-3-small"


def _get_embeddings() -> OpenAIEmbeddings:
    """Return a configured OpenAIEmbeddings instance."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def process_new_documents(file_path: str) -> int:
    """
    Ingest a PDF file into the local FAISS vector store.

    Steps:
      1. Load all pages from the PDF.
      2. Split pages into overlapping text chunks.
      3. Embed each chunk with OpenAI.
      4. Merge into (or create) the persistent FAISS index.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the PDF file to ingest.

    Returns
    -------
    int
        Number of text chunks added to the vector store.

    Raises
    ------
    FileNotFoundError
        If the supplied file_path does not exist.
    ValueError
        If the PDF contains no extractable text.
    """

    # ── 1. Validate input ─────────────────────────────────────────────────────
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    # ── 2. Load the PDF ───────────────────────────────────────────────────────
    loader = PyPDFLoader(file_path)
    raw_pages = loader.load()           # List[Document], one per page

    if not raw_pages:
        raise ValueError(f"No text could be extracted from: {file_path}")

    # ── 3. Split into chunks ──────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Keep sentences intact where possible
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_pages)

    if not chunks:
        raise ValueError("Text splitter produced zero chunks — the PDF may be empty.")

    # ── 4. Embed & persist ────────────────────────────────────────────────────
    embeddings = _get_embeddings()
    db_path    = Path(FAISS_DB_PATH)

    if db_path.exists() and any(db_path.iterdir()):
        # ── 4a. Existing DB: load and merge ───────────────────────────────────
        # allow_dangerous_deserialization is required when loading pickle-based
        # FAISS indexes from disk; safe because we created the index locally.
        vector_store = FAISS.load_local(
            folder_path=str(db_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store.add_documents(chunks)
        print(f"[ingestion] Merged {len(chunks)} new chunks into existing FAISS DB.")
    else:
        # ── 4b. No DB yet: create a fresh one ─────────────────────────────────
        db_path.mkdir(parents=True, exist_ok=True)
        vector_store = FAISS.from_documents(chunks, embeddings)
        print(f"[ingestion] Created new FAISS DB with {len(chunks)} chunks.")

    # Persist the (updated) index to disk
    vector_store.save_local(str(db_path))
    print(f"[ingestion] FAISS DB saved to '{db_path}'.")

    return len(chunks)


# ── Quick sanity-check when run directly ──────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <path/to/file.pdf>")
        sys.exit(1)

    added = process_new_documents(sys.argv[1])
    print(f"Done — {added} chunks ingested.")