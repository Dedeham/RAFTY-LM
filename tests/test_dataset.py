import copy
from collections import Counter

import pytest

from src.dataset import generate_datasets, format_doc
from src.store import format_doc as store_format_doc

TEST_CFG = {
    "corpus": {"doc_char_limit": 450},
    "dataset": {
        "n_distractors": 3,
        "p_golden": 0.85,
        "p_unanswerable": 0.15,
        "hard_negative_ratio": 0.5,
        "seed": 42,
        "domain_label": "test domain",
    },
}

POOL_SIZE_PER_SPLIT = 60
QA_COUNT = 2000


class FakeDoc:
    def __init__(self, metadata):
        self.metadata = metadata


class FakePollutedRetriever:
    def __init__(self, all_chunks):
        self.all_chunks = all_chunks

    def invoke(self, question):
        return [FakeDoc({"chunk_id": c["chunk_id"]}) for c in self.all_chunks]


def make_chunk_pool(prefix, split, count):
    return [
        {
            "chunk_id": f"{prefix}_{split}_{i}",
            "source_file": f"{prefix}.pdf",
            "page_number": i + 1,
            "split": split,
            "text": f"Unique synthetic marker {prefix}_{split}_{i} torque value {i} N*m.",
        }
        for i in range(count)
    ]


def make_qa_bank(chunks, count):
    return [
        {
            "chunk_id": chunks[i % len(chunks)]["chunk_id"],
            "split": chunks[i % len(chunks)]["split"],
            "question": f"What is the torque value {i}?",
            "answer": f"Answer {i}",
            "key_fact": chunks[i % len(chunks)]["text"][:20],
        }
        for i in range(count)
    ]


@pytest.fixture
def synthetic_env(tmp_path, monkeypatch):
    train_chunks = make_chunk_pool("train_doc", "train", POOL_SIZE_PER_SPLIT)
    eval_chunks = make_chunk_pool("eval_doc", "eval", POOL_SIZE_PER_SPLIT)
    all_chunks = train_chunks + eval_chunks
    qa_bank = make_qa_bank(all_chunks, QA_COUNT)

    monkeypatch.setattr("src.dataset.load_config", lambda: copy.deepcopy(TEST_CFG))
    monkeypatch.setattr("src.dataset.load_qa_bank", lambda path=None: copy.deepcopy(qa_bank))
    monkeypatch.setattr("src.dataset.get_chunks", lambda config=None: copy.deepcopy(all_chunks))
    monkeypatch.setattr("src.dataset.get_retriever", lambda k=None, config=None: FakePollutedRetriever(all_chunks))
    monkeypatch.setattr("src.dataset.OUTPUT_TRAIN", tmp_path / "raft_train.json")
    monkeypatch.setattr("src.dataset.OUTPUT_EVAL", tmp_path / "raft_eval.json")

    return all_chunks


def chunk_ids_referenced(samples, all_chunk_ids):
    referenced = set()
    for sample in samples:
        blob = "\n".join(sample["context"])
        for chunk_id in all_chunk_ids:
            if chunk_id in blob:
                referenced.add(chunk_id)
    return referenced


def test_no_split_leakage(synthetic_env):
    all_chunks = synthetic_env
    all_chunk_ids = [c["chunk_id"] for c in all_chunks]

    result = generate_datasets()

    train_referenced = chunk_ids_referenced(result["train"], all_chunk_ids)
    eval_referenced = chunk_ids_referenced(result["eval"], all_chunk_ids)

    assert train_referenced
    assert eval_referenced
    assert train_referenced.isdisjoint(eval_referenced)
    assert all(cid.startswith("train_doc_train_") for cid in train_referenced)
    assert all(cid.startswith("eval_doc_eval_") for cid in eval_referenced)


def test_deterministic_with_fixed_seed(synthetic_env):
    result_a = generate_datasets()
    result_b = generate_datasets()

    assert result_a == result_b


def test_status_ratios_within_tolerance(synthetic_env):
    result = generate_datasets()
    all_samples = result["train"] + result["eval"]
    counts = Counter(s["status"] for s in all_samples)
    total = len(all_samples)

    p_golden = TEST_CFG["dataset"]["p_golden"]
    p_unanswerable = TEST_CFG["dataset"]["p_unanswerable"]
    expected_answer_found = (1 - p_unanswerable) * p_golden

    actual_answer_found = counts["answer_found"] / total

    assert abs(actual_answer_found - expected_answer_found) < 0.05


def test_format_doc_matches_shared_store_format():
    chunk = {"source_file": "manual.pdf", "page_number": 12, "text": "Torque spec is 45 N*m."}

    assert format_doc(3, chunk) == store_format_doc(3, chunk["source_file"], chunk["page_number"], chunk["text"])
    assert format_doc(3, chunk) == "[Doc 3 | Page 12 | Source: manual.pdf]: Torque spec is 45 N*m."
