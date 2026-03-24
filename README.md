<div align="center">
  
# Enterprise Document Intelligence (RAG)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Premium_UI-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-black.svg)](https://openai.com/)

**A production-ready Retrieval-Augmented Generation (RAG) application featuring local FAISS vector storage, secure PDF ingestion, and a custom enterprise-grade user interface.**

</div>

---

##  Overview

This application provides a secure, hallucination-free chat interface for your documents. Users can upload multiple PDFs, which are locally chunked, embedded using OpenAI's `text-embedding-3-small`, and stored in a persistent FAISS vector database. The chatbot utilizes `gpt-4o` to answer questions strictly based on the uploaded context, providing exact source citations for every answer.

##  Key Features

* **Premium UI/UX:** A highly customized Streamlit interface designed to look like a modern SaaS application.
* **Local Vector Storage:** Uses FAISS to store embeddings locally, ensuring quick retrieval without recurring database hosting costs.
* **Streaming Responses:** Answers are streamed token-by-token for a fast, conversational feel.
* **Source Citations:** Every AI response includes expandable citations showing the exact excerpt and page number used to generate the answer.
* **Memory & Context:** Maintains full conversation history, allowing for follow-up questions and multi-turn reasoning.

---

## 🏗️ Project Structure

* `app.py` - The main Streamlit frontend containing the UI, chat state management, and LangChain retrieval logic.
* `ingestion.py` - The backend processing engine that handles PDF loading, recursive character text splitting, and FAISS database operations.
* `.env` - (Not tracked) Stores the `OPENAI_API_KEY`.
* `docs/` - (Not tracked) Temporary staging directory for uploaded PDFs.
* `db/` - (Not tracked) Directory where the persistent FAISS index is saved.

---

## 🛠️ Installation & Setup

### 1. Clone & Environment
Clone the repository and set up a virtual environment to keep dependencies isolated:

```bash
git clone [https://github.com/YourUsername/Enterprise-RAG-Chatbot.git](https://github.com/AbdullahMJamal/Enterprise-RAG-Chatbot.git)
cd Enterprise-RAG-Chatbot

python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
````

### 2\. Install Dependencies

Install the required packages using `pip`:

```bash
pip install streamlit langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu pypdf python-dotenv
```

### 3\. Environment Variables

Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY="sk-your-api-key-here"
```

-----

##  Running the Application

Once your environment is set up, launch the Streamlit server:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

### Usage Flow:

1.  Open the sidebar and upload a PDF document.
2.  Wait for the ingestion progress bar to complete (extracting, embedding, saving to FAISS).
3.  Type a question into the chat input.
4.  Review the AI's answer and click "View Context Sources" to see the extracted text.

-----
"# Enterprise-RAG-Chatbot" 
"# Enterprise-RAG-Chatbot" 
