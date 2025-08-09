# Local PDF Q&A Chatbot (LangChain + FAISS + Ollama)

A simple Streamlit RAG chatbot that indexes your PDF into FAISS and answers questions using a small local LLM via Ollama.

## Features
- Choose embeddings backend: **Ollama embeddings** (recommended: `nomic-embed-text`) or **Hugging Face** (`all-MiniLM-L6-v2`)
- Vector store: **FAISS**
- Local LLM via Ollama: default **`phi3:mini`**
- Source documents shown per answer

## Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- Pull models you plan to use:
  ```bash
  ollama pull phi3:mini
  ollama pull nomic-embed-text   # if using Ollama embeddings
  ```

## Setup
If you can create virtualenvs:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your environment is locked down (PEP 668), run locally on your machine instead, or use a venv with `python3-venv` installed.

## Run
```bash
streamlit run app.py
```
Open the URL shown in the terminal. Upload your PDF, click "Build / Load Index", then start chatting.

## Notes
- For Ollama embeddings, select "Ollama" in the sidebar and ensure `nomic-embed-text` is pulled.
- For Hugging Face embeddings, select "HuggingFace" and ensure Python deps are installed.
- Indexes are cached under `faiss_indexes/` per-PDF and embedding configuration.
