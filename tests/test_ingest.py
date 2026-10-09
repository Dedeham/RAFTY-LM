from pathlib import Path

from langchain_core.documents import Document

import src.ingest as ingest
from src.ingest import load_and_chunk_documents, _chunk_split


class StubLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        return [
            Document(page_content="Front caliper torque is 123 N*m.", metadata={"page": 0}),
            Document(page_content="Rear caliper torque is 98 N*m.", metadata={"page": 1}),
        ]


def test_chunk_split_boundaries():
    chunk_id = "any_chunk_id_0"
    assert _chunk_split(chunk_id, 0.0) == "train"
    assert _chunk_split(chunk_id, 1.0) == "eval"


def test_load_and_chunk_documents_metadata_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PyMuPDFLoader", StubLoader)
    (tmp_path / "dummy.pdf").write_bytes(b"")

    chunks = load_and_chunk_documents(
        pdf_directory=tmp_path,
        chunk_size=1000,
        chunk_overlap=0,
        eval_split=0.0,
    )

    assert len(chunks) == 2
    expected_keys = {"source_file", "page_number", "chunk_id", "split"}
    for idx, chunk in enumerate(chunks):
        assert set(chunk.metadata.keys()) == expected_keys
        assert chunk.metadata["source_file"] == "dummy.pdf"
        assert chunk.metadata["chunk_id"] == f"dummy.pdf_chunk_{idx}"
        assert chunk.metadata["split"] == "train"

    assert chunks[0].metadata["page_number"] == 1
    assert chunks[1].metadata["page_number"] == 2


def test_load_and_chunk_documents_raises_on_empty_directory(tmp_path):
    try:
        load_and_chunk_documents(pdf_directory=tmp_path, chunk_size=1000, chunk_overlap=0, eval_split=0.0)
        assert False, "expected ValueError for empty pdf directory"
    except ValueError:
        pass
