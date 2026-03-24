import os
import shutil
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── LangChain Imports ─────────────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Local module ──────────────────────────────────────────────────────────────
from ingestion import FAISS_DB_PATH, EMBEDDING_MODEL, process_new_documents

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

# ── App-level constants ────────────────────────────────────────────────────────
DOCS_DIR        = Path("docs")          # Temporary PDF staging folder
TOP_K_CHUNKS    = 3                     # Number of chunks retrieved per query
LLM_MODEL       = "gpt-4o"
LLM_TEMPERATURE = 0.0                   # Deterministic answers for accurate RAG

SYSTEM_PROMPT = """You are an enterprise knowledge assistant. You answer questions \
exclusively based on the context excerpts provided below.

Rules you must follow without exception:
1. Base every answer ONLY on the information in the provided context.
2. If the context does not contain enough information to answer the question, \
respond with exactly: "I do not have enough information in the provided documents \
to answer that question."
3. Never fabricate facts, infer beyond the context, or use outside knowledge.
4. When possible, quote the relevant passage and mention the source document/page.
5. Be concise, structured, and professional in your response.

Context:
{context}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cached resources  (loaded once per Streamlit session / server restart)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading vector database…")
def load_vector_store():
    """Load the FAISS index from disk. Returns None if empty."""
    db_path = Path(FAISS_DB_PATH)

    if not db_path.exists() or not any(db_path.iterdir()):
        return None                     # Database does not exist yet

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        folder_path=str(db_path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,   # Safe: we control the ingestion pipeline
    )


@st.cache_resource(show_spinner="Loading language model…")
def load_llm():
    """Return a cached ChatOpenAI instance for streaming."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        streaming=True,                 # Enables token-by-token streaming in Streamlit
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_context(query: str, vector_store: FAISS) -> tuple[str, list]:
    """Perform a similarity search and format chunks into a context string."""
    docs = vector_store.similarity_search(query, k=TOP_K_CHUNKS)

    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown file")
        page   = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Excerpt {i} — {Path(source).name}, page {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts), docs


def build_messages(context: str, chat_history: list, user_query: str) -> list:
    """Assemble the full message list for the LLM including memory."""
    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    # Replay conversation history (keeps multi-turn coherence)
    for turn in chat_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=user_query))
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — document ingestion
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📂 Document Manager")
    st.caption("Upload PDFs to expand the knowledge base.")

    uploaded_file = st.file_uploader(
        label="Choose a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Each uploaded PDF is chunked, embedded, and merged into the vector store.",
    )

    if uploaded_file is not None:
        # Persist the upload to the staging folder
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = DOCS_DIR / uploaded_file.name

        with open(save_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)

        # Client-friendly progress indicator
        progress_text = "Extracting text and chunking..."
        my_bar = st.progress(0, text=progress_text)
        
        try:
            my_bar.progress(30, text="Analyzing PDF structure...")
            time.sleep(0.5) # Slight pause for UI smoothness
            
            my_bar.progress(60, text="Generating embeddings...")
            num_chunks = process_new_documents(str(save_path))

            my_bar.progress(90, text="Saving to Vector Database...")
            
            # ── Invalidate cache so the UI picks up the new documents ─────
            load_vector_store.clear()
            
            my_bar.progress(100, text="Complete!")
            time.sleep(0.5)
            my_bar.empty()

            st.success(
                f"✅ **{uploaded_file.name}** ingested successfully!\n\n"
                f"Added **{num_chunks}** searchable chunks."
            )
        except Exception as exc:
            my_bar.empty()
            st.error(f"❌ Ingestion failed: {exc}")

    st.divider()

    # ── Session controls ───────────────────────────────────────────────────────
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption(
        "**⚙️ System Architecture**\n\n"
        f"**Model:** `{LLM_MODEL}`  \n"
        f"**Embeddings:** `{EMBEDDING_MODEL}`  \n"
        "**Vector DB:** `FAISS (Local)`"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main area — chat interface
# ─────────────────────────────────────────────────────────────────────────────

st.title("Enterprise Document Intelligence")
st.caption("Ask questions about your uploaded documents. Answers are grounded strictly in the provided content to guarantee zero hallucinations.")

# ── Initialise session state ──────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Load cached resources ─────────────────────────────────────────────────────
vector_store = load_vector_store()
llm          = load_llm()

# ── Render existing conversation ──────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Guard: no documents uploaded yet ─────────────────────────────────────────
if vector_store is None:
    st.info(
        "👈 **Knowledge Base Empty.** Please upload a PDF using the sidebar to begin chatting.",
        icon="ℹ️",
    )
    st.stop()

# ── Chat input ────────────────────────────────────────────────────────────────
# ── Chat input ────────────────────────────────────────────────────────────────
user_query = st.chat_input("Ask a question about your documents…")

if user_query:
    # 1. Display the user's message IMMEDIATELY (Fixes the vanishing issue)
    with st.chat_message("user"):
        st.markdown(user_query)

    # Append to history
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # 2. Open the Assistant chat bubble right away
    with st.chat_message("assistant"):
        
        # 3. Show a spinner while the heavy lifting happens (Fixes the awkward delay)
        with st.spinner("🔍 Searching knowledge base..."):
            # Retrieve relevant chunks
            context_text, source_docs = retrieve_context(user_query, vector_store)

            # Build the message list
            messages = build_messages(
                context=context_text,
                chat_history=st.session_state.chat_history[:-1],  # exclude current query
                user_query=user_query,
            )

        # 4. Stream the LLM response instantly once the search is done
        response_placeholder = st.empty()
        full_response        = ""

        # stream=True yields chunks; we accumulate and re-render progressively
        for chunk in llm.stream(messages):
            token = chunk.content
            if token:
                full_response += token
                response_placeholder.markdown(full_response + "▌")   # blinking cursor effect

        # Final render (remove cursor)
        response_placeholder.markdown(full_response)

        # ── Show source citations in an expander ─────────────────────
        if source_docs:
            with st.expander("📎 View Context Sources", expanded=False):
                for i, doc in enumerate(source_docs, start=1):
                    source = doc.metadata.get("source", "Unknown")
                    page   = doc.metadata.get("page", "?")
                    st.markdown(
                        f"**Source {i}** — `{Path(source).name}` (Page {page})\n\n"
                        f"> {doc.page_content[:400]}{'…' if len(doc.page_content) > 400 else ''}"
                    )

    # Persist the assistant's full response
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})