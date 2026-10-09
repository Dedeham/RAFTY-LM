import json
import random
from collections import Counter
from pathlib import Path

from src.config import load_config
from src.store import get_chunks, get_retriever, format_doc as store_format_doc

QA_BANK_PATH = Path("data/qa_bank.json")
OUTPUT_TRAIN = Path("data/raft_train.json")
OUTPUT_EVAL = Path("data/raft_eval.json")

UNANSWERABLE_MESSAGE = "The provided context excerpts do not contain the necessary information or specifications to answer this question."


def load_qa_bank(path=QA_BANK_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chunk_lookup(doc_char_limit, config=None):
    chunks = get_chunks(config=config)
    chunk_by_id = {}
    chunks_by_split = {"train": [], "eval": []}
    for chunk in chunks:
        chunk = dict(chunk)
        chunk["text"] = chunk["text"][:doc_char_limit]
        chunk_by_id[chunk["chunk_id"]] = chunk
        if chunk["split"] in chunks_by_split:
            chunks_by_split[chunk["split"]].append(chunk)
    return chunk_by_id, chunks_by_split


def build_retriever_state(k, config=None):
    state = {"retriever": None, "warned": False}
    try:
        state["retriever"] = get_retriever(k=k, config=config)
    except Exception as e:
        print(f"Warning: retriever unavailable ({e}); using random distractor sampling only.")
        state["warned"] = True
    return state


def get_hard_negatives(question, split, exclude_id, n, retriever_state, chunk_by_id):
    if n <= 0 or retriever_state["retriever"] is None:
        return []
    try:
        docs = retriever_state["retriever"].invoke(question)
    except Exception as e:
        if not retriever_state["warned"]:
            print(f"Warning: hard-negative retrieval failed ({e}); falling back to random distractors.")
            retriever_state["warned"] = True
        retriever_state["retriever"] = None
        return []

    picked = []
    seen = set()
    for doc in docs:
        meta = doc.metadata or {}
        chunk_id = meta.get("chunk_id")
        if not chunk_id or chunk_id == exclude_id or chunk_id in seen:
            continue
        chunk = chunk_by_id.get(chunk_id)
        if not chunk or chunk["split"] != split:
            continue
        picked.append(chunk)
        seen.add(chunk_id)
        if len(picked) >= n:
            break
    return picked


def sample_random_distractors(pool, exclude_ids, n):
    candidates = [c for c in pool if c["chunk_id"] not in exclude_ids]
    return random.sample(candidates, min(n, len(candidates)))


def sample_distractors(question, split, exclude_id, n, pool, retriever_state, chunk_by_id, hard_negative_ratio):
    hard_n = round(n * hard_negative_ratio)
    hard = get_hard_negatives(question, split, exclude_id, hard_n, retriever_state, chunk_by_id)
    used_ids = {exclude_id} | {c["chunk_id"] for c in hard}
    remaining = n - len(hard)
    random_picks = sample_random_distractors(pool, used_ids, remaining)
    return hard + random_picks


def format_doc(index, chunk):
    return store_format_doc(index, chunk["source_file"], chunk["page_number"], chunk["text"])


def build_answer_found_sample(question, answer, key_fact, golden, distractors):
    combined = list(distractors)
    golden_index = random.randint(0, len(combined))
    combined.insert(golden_index, golden)

    context_docs = [format_doc(i, c) for i, c in enumerate(combined)]

    cot_lines = [
        f"Analyzing retrieved context documents for the query: '{question}'",
        "Document evaluation:",
    ]
    for i, c in enumerate(combined):
        if i == golden_index:
            cot_lines.append(
                f"- Doc {i} [Page {c['page_number']} | Source: {c['source_file']}]: Directly addresses the question. Key fact: {key_fact}."
            )
        else:
            cot_lines.append(
                f"- Doc {i} [Page {c['page_number']} | Source: {c['source_file']}]: Irrelevant distractor."
            )
    cot_lines.append(f"Conclusion: Based strictly on Doc {golden_index}, {answer}")

    return {
        "question": question,
        "context": context_docs,
        "chain_of_thought": "\n".join(cot_lines),
        "status": "answer_found",
        "answer": answer,
    }


def build_all_distractor_sample(question, distractors):
    context_docs = [format_doc(i, c) for i, c in enumerate(distractors)]

    cot_lines = [
        f"Analyzing retrieved context documents for the query: '{question}'",
        "Document evaluation:",
    ]
    for i, c in enumerate(distractors):
        cot_lines.append(
            f"- Doc {i} [Page {c['page_number']} | Source: {c['source_file']}]: Contains no information regarding the question."
        )
    cot_lines.append(
        f"Conclusion: None of the {len(distractors)} provided documents (Doc 0 through Doc {len(distractors) - 1}) contain data addressing this question. All candidate documents are irrelevant distractors."
    )

    return {
        "question": question,
        "context": context_docs,
        "chain_of_thought": "\n".join(cot_lines),
        "status": "all_distractors",
        "answer": UNANSWERABLE_MESSAGE,
    }


def build_sample(qa, chunk_by_id, chunks_by_split, retriever_state, dataset_cfg):
    split = qa["split"]
    golden = chunk_by_id.get(qa["chunk_id"])
    pool = chunks_by_split.get(split, [])
    question = qa["question"]
    n_distractors = dataset_cfg["n_distractors"]
    hard_negative_ratio = dataset_cfg["hard_negative_ratio"]

    if golden is None or random.random() < dataset_cfg["p_unanswerable"]:
        exclude_id = golden["chunk_id"] if golden else None
        distractors = sample_distractors(
            question, split, exclude_id, n_distractors + 1, pool, retriever_state, chunk_by_id, hard_negative_ratio
        )
        return build_all_distractor_sample(question, distractors)

    if random.random() < dataset_cfg["p_golden"]:
        distractors = sample_distractors(
            question, split, golden["chunk_id"], n_distractors, pool, retriever_state, chunk_by_id, hard_negative_ratio
        )
        return build_answer_found_sample(question, qa["answer"], qa["key_fact"], golden, distractors)

    distractors = sample_distractors(
        question, split, golden["chunk_id"], n_distractors + 1, pool, retriever_state, chunk_by_id, hard_negative_ratio
    )
    return build_all_distractor_sample(question, distractors)


def generate_datasets(config=None):
    cfg = config or load_config()
    dataset_cfg = cfg["dataset"]
    random.seed(dataset_cfg["seed"])

    doc_char_limit = cfg["corpus"]["doc_char_limit"]
    qa_bank = load_qa_bank()
    chunk_by_id, chunks_by_split = build_chunk_lookup(doc_char_limit, config=cfg)

    retrieval_k = max(30, (dataset_cfg["n_distractors"] + 1) * 10)
    retriever_state = build_retriever_state(retrieval_k, config=cfg)

    samples_by_split = {"train": [], "eval": []}
    for qa in qa_bank:
        split = qa.get("split")
        if split not in samples_by_split:
            continue
        sample = build_sample(qa, chunk_by_id, chunks_by_split, retriever_state, dataset_cfg)
        samples_by_split[split].append(sample)

    OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        json.dump(samples_by_split["train"], f, indent=2, ensure_ascii=False)
    with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
        json.dump(samples_by_split["eval"], f, indent=2, ensure_ascii=False)

    train_status = Counter(s["status"] for s in samples_by_split["train"])
    eval_status = Counter(s["status"] for s in samples_by_split["eval"])
    print(f"Train samples: {len(samples_by_split['train'])} -> {dict(train_status)}")
    print(f"Eval samples: {len(samples_by_split['eval'])} -> {dict(eval_status)}")

    return samples_by_split


if __name__ == "__main__":
    generate_datasets()
