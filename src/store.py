import re
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.config import load_config
from src.ingest import load_and_chunk_documents


def get_embedding_function(config=None):
    cfg = config or load_config()
    return OllamaEmbeddings(model=cfg["corpus"]["embed_model"])


def build_vector_store(pdf_directory=None, persist_directory=None, config=None):
    cfg = config or load_config()
    corpus_cfg = cfg["corpus"]
    persist_directory = persist_directory or corpus_cfg["chroma_path"]
    collection_name = corpus_cfg["collection"]

    chunks = load_and_chunk_documents(pdf_directory=pdf_directory, config=cfg)

    print(f"Initializing Ollama '{corpus_cfg['embed_model']}' embeddings...")
    embeddings = get_embedding_function(cfg)

    existing = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name=collection_name)
    existing.delete_collection()

    print(f"Computing vectors for {len(chunks)} chunks and saving to '{persist_directory}'...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("Done. Vector store saved successfully.")
    return vector_db


def get_retriever(k=None, persist_directory=None, config=None):
    cfg = config or load_config()
    corpus_cfg = cfg["corpus"]
    persist_directory = persist_directory or corpus_cfg["chroma_path"]
    k = k if k is not None else cfg["agent"]["k"]

    embeddings = get_embedding_function(cfg)
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=corpus_cfg["collection"]
    )
    return vector_db.as_retriever(search_kwargs={"k": k})


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def format_doc(index, source_file, page_number, text):
    return f"[Doc {index} | Page {page_number} | Source: {source_file}]: {text}"


def get_chunks(split=None, persist_directory=None, config=None):
    cfg = config or load_config()
    corpus_cfg = cfg["corpus"]
    persist_directory = persist_directory or corpus_cfg["chroma_path"]

    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embedding_function(cfg),
        collection_name=corpus_cfg["collection"]
    )
    data = vector_db.get(include=["metadatas", "documents"])

    chunks = []
    for meta, text in zip(data["metadatas"], data["documents"]):
        if split is not None and meta.get("split") != split:
            continue
        chunks.append({
            "chunk_id": meta.get("chunk_id"),
            "source_file": meta.get("source_file"),
            "page_number": meta.get("page_number"),
            "split": meta.get("split"),
            "text": normalize_whitespace(text),
        })
    return chunks


if __name__ == "__main__":
    build_vector_store()
