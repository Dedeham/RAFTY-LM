import hashlib
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import load_config


def _chunk_split(chunk_id, eval_split):
    bucket = int(hashlib.sha1(chunk_id.encode()).hexdigest(), 16) % 100
    return "eval" if bucket < eval_split * 100 else "train"


def load_and_chunk_documents(pdf_directory=None, chunk_size=None, chunk_overlap=None, eval_split=None, config=None):
    cfg = config or load_config()
    corpus_cfg = cfg["corpus"]
    pdf_directory = Path(pdf_directory) if pdf_directory is not None else Path(corpus_cfg["pdf_dir"])
    chunk_size = chunk_size if chunk_size is not None else corpus_cfg["chunk_size"]
    chunk_overlap = chunk_overlap if chunk_overlap is not None else corpus_cfg["chunk_overlap"]
    eval_split = eval_split if eval_split is not None else corpus_cfg["eval_split"]

    pdf_files = sorted(pdf_directory.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_directory.resolve()}")

    print(f"Found {len(pdf_files)} PDF files. Reading pages...")
    raw_docs = []

    for pdf_path in pdf_files:
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            docs = loader.load()
            print(f"  [Read] {pdf_path.name}: {len(docs)} pages")

            for d in docs:
                page_num = d.metadata.get("page", 0) + 1
                d.metadata = {
                    "source_file": pdf_path.name,
                    "page_number": page_num,
                }
            raw_docs.extend(docs)
        except Exception as e:
            print(f"  [Error reading {pdf_path.name}]: {e}")

    print(f"Successfully read {len(raw_docs)} pages.")
    print(f"Splitting pages into {chunk_size}-character chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(raw_docs)

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{chunk.metadata.get('source_file', 'doc')}_chunk_{idx}"
        chunk.metadata = {
            "source_file": chunk.metadata.get("source_file", "doc"),
            "page_number": chunk.metadata.get("page_number", 0),
            "chunk_id": chunk_id,
            "split": _chunk_split(chunk_id, eval_split),
        }

    print(f"Successfully created {len(chunks)} chunks.")
    return chunks


if __name__ == "__main__":
    load_and_chunk_documents()
