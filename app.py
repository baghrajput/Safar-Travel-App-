import os
import io
import json
import hashlib
from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA


APP_TITLE = "Local PDF Q&A Chatbot (LangChain + FAISS + Ollama)"
DEFAULT_EMBED_BACKEND = "Ollama"  # Ollama or HuggingFace
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"  # ollama pull nomic-embed-text
DEFAULT_HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "phi3:mini"  # Small and fast; works on CPU
INDEX_ROOT = Path("./faiss_indexes")
INDEX_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------

def compute_pdf_fingerprint(file_path: Path, embed_backend: str, embed_model_name: str) -> str:
    stat = file_path.stat()
    hasher = hashlib.sha256()
    hasher.update(str(file_path.resolve()).encode("utf-8"))
    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    hasher.update(embed_backend.encode("utf-8"))
    hasher.update(embed_model_name.encode("utf-8"))
    return hasher.hexdigest()[:16]


def get_index_dir_for_pdf(file_path: Path, embed_backend: str, embed_model_name: str) -> Path:
    fp = compute_pdf_fingerprint(file_path, embed_backend, embed_model_name)
    return INDEX_ROOT / f"index_{file_path.stem}_{embed_backend}_{fp}"


def save_index_metadata(index_dir: Path, metadata: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_index_metadata(index_dir: Path) -> dict | None:
    meta_path = index_dir / "metadata.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def build_embeddings(backend: str, model_name: str):
    if backend == "Ollama":
        return OllamaEmbeddings(model=model_name)
    # Default to HF CPU embeddings
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})


def load_or_build_vectorstore(pdf_path: Path, embed_backend: str, embed_model_name: str) -> FAISS:
    embeddings = build_embeddings(embed_backend, embed_model_name)
    index_dir = get_index_dir_for_pdf(pdf_path, embed_backend, embed_model_name)

    faiss_index_path = index_dir / "index.faiss"
    faiss_store_path = index_dir / "index.pkl"

    if faiss_index_path.exists() and faiss_store_path.exists():
        try:
            vs = FAISS.load_local(
                folder_path=str(index_dir),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            return vs
        except Exception:
            pass  # fall through to rebuild

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(index_dir))

    save_index_metadata(index_dir, {
        "pdf_path": str(pdf_path.resolve()),
        "embedding_backend": embed_backend,
        "embedding_model": embed_model_name,
        "num_chunks": len(chunks),
    })

    return vs


def make_retriever(vectorstore: FAISS, k: int = 5):
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def make_llm(model_name: str, temperature: float):
    return ChatOllama(model=model_name, temperature=temperature)


def make_qa_chain(llm: ChatOllama, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )


# -----------------------------
# Streamlit App
# -----------------------------

def init_session_state():
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_pdf_path" not in st.session_state:
        st.session_state.current_pdf_path = None


def sidebar_controls():
    st.sidebar.subheader("Settings")

    embed_backend = st.sidebar.radio(
        "Embedding backend",
        options=["Ollama", "HuggingFace"],
        index=0 if DEFAULT_EMBED_BACKEND == "Ollama" else 1,
        help="Ollama embeddings avoid heavy local installs (pull an embedding model like nomic-embed-text).",
    )

    if embed_backend == "Ollama":
        embed_model = st.sidebar.text_input(
            "Ollama embedding model",
            value=DEFAULT_OLLAMA_EMBED_MODEL,
            help="Run `ollama pull nomic-embed-text` or use another embedding model available in Ollama.",
        )
    else:
        embed_model = st.sidebar.text_input(
            "HF embedding model",
            value=DEFAULT_HF_EMBED_MODEL,
            help="Hugging Face embedding model used for FAISS. Requires sentence-transformers.",
        )

    ollama_model = st.sidebar.text_input(
        "Ollama chat model",
        value=DEFAULT_OLLAMA_MODEL,
        help="Run `ollama pull <model>` first. Examples: phi3:mini, llama3, mistral:instruct",
    )

    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.05)
    k = st.sidebar.slider("Top-K passages", 1, 10, 5)

    return embed_backend, embed_model, ollama_model, temperature, k


def render_chat():
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Ask a question about your PDF...")
    if not user_input:
        return

    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.qa_chain is None:
        with st.chat_message("assistant"):
            st.error("Please upload a PDF and build the index first.")
        return

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.qa_chain({"query": user_input})
                answer = result.get("result") or result.get("answer") or "(no answer)"
                sources = result.get("source_documents", [])
            except Exception as e:
                answer = f"Error while generating answer: {e}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("Show sources"):
                for idx, doc in enumerate(sources, start=1):
                    meta = doc.metadata or {}
                    page = meta.get("page", "?")
                    st.markdown(f"- Source {idx}: page {page}")

        st.session_state.chat_history.append(("assistant", answer))


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")
    init_session_state()

    st.title(APP_TITLE)
    st.caption("RAG with FAISS, local embeddings (Ollama or HF), and a small local LLM via Ollama.")

    embed_backend, embed_model, ollama_model, temperature, k = sidebar_controls()

    st.markdown("**1) Upload your PDF**")
    upload = st.file_uploader("PDF with Q&A or knowledge content", type=["pdf"])

    col1, col2 = st.columns([1, 1])
    with col1:
        build_clicked = st.button("Build / Load Index", type="primary")
    with col2:
        clear_clicked = st.button("Clear Chat")

    if clear_clicked:
        st.session_state.chat_history = []

    if build_clicked:
        if upload is None:
            st.warning("Please upload a PDF first.")
        else:
            # Persist upload to a temp file
            tmp_dir = Path(st.secrets.get("tmp_dir", "/tmp"))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = tmp_dir / upload.name
            with open(pdf_path, "wb") as f:
                f.write(upload.getbuffer())

            with st.spinner("Building or loading FAISS index..."):
                try:
                    vectorstore = load_or_build_vectorstore(pdf_path, embed_backend, embed_model)
                    retriever = make_retriever(vectorstore, k=k)
                    llm = make_llm(ollama_model, temperature)
                    qa = make_qa_chain(llm, retriever)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.qa_chain = qa
                    st.session_state.current_pdf_path = str(pdf_path)

                    index_dir = get_index_dir_for_pdf(pdf_path, embed_backend, embed_model)
                    meta = load_index_metadata(index_dir) or {}
                    st.success(
                        f"Index ready. Chunks: {meta.get('num_chunks', 'unknown')} | Embeddings: {embed_backend}:{embed_model} | Model: {ollama_model}"
                    )
                except Exception as e:
                    st.error(f"Failed to build/load index: {e}")

    st.markdown("---")
    st.markdown("**2) Chat**")

    if st.session_state.qa_chain is None:
        st.info("Upload a PDF and click 'Build / Load Index' to start chatting.")

    render_chat()


if __name__ == "__main__":
    main()