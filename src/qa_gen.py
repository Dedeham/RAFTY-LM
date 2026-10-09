import json
import os
import re
from pathlib import Path

from src.config import load_config
from src.store import get_chunks
from src.jsonutil import strip_json_fences

QA_BANK_PATH = Path("data/qa_bank.json")
MAX_ANSWER_CHARS = 500
SELF_REFERENTIAL_PHRASES = ["document", "manual", "text", "passage", "chunk", "according to"]

PROMPT_TEMPLATE = """You are creating a training example for a question-answering system.
Given the passage below, write ONE question that a real user could naturally ask, and its answer, based ONLY on the information in the passage.

Rules:
- The question must stand on its own. NEVER use words like "document", "manual", "text", "passage", "chunk", or phrases like "according to" -- for example, do not write "What does the document say about X", write "What is X" instead.
- The answer must be fully supported by the passage below and nothing else.
- "key_fact" must be a short quote copied verbatim from the passage that proves the answer.

Output ONLY a single JSON object with exactly these keys, no other text:
{{"question": "...", "answer": "...", "key_fact": "..."}}

Passage:
{chunk_text}"""


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def looks_like_index(text):
    return len(re.findall(r"\.{3,}", text)) >= 5


def build_llm(qa_cfg):
    backend = qa_cfg["backend"]
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=qa_cfg["model"])
    if backend == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for the openai qa backend")
        return ChatOpenAI(model=qa_cfg["model"], base_url=qa_cfg.get("base_url"), api_key=api_key)
    raise ValueError(f"Unsupported qa backend: {backend}")


def build_prompt(chunk_text):
    return PROMPT_TEMPLATE.format(chunk_text=chunk_text)


def parse_qa_response(content):
    return json.loads(strip_json_fences(content))


def validate_qa(item, chunk_text):
    question = item.get("question", "")
    answer = item.get("answer", "")
    key_fact = item.get("key_fact", "")
    if not question or not answer or not key_fact:
        return False
    if len(answer) > MAX_ANSWER_CHARS:
        return False
    if any(phrase in question.lower() for phrase in SELF_REFERENTIAL_PHRASES):
        return False
    if normalize(key_fact) not in normalize(chunk_text):
        return False
    return True


def generate_one_qa(llm, chunk_text):
    for _ in range(2):
        response = llm.invoke(build_prompt(chunk_text))
        try:
            item = parse_qa_response(response.content)
        except Exception:
            continue
        if validate_qa(item, chunk_text):
            return item
    return None


def load_existing_bank():
    if QA_BANK_PATH.exists():
        with open(QA_BANK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def generate_qa_bank(config=None):
    cfg = config or load_config()
    qa_cfg = cfg["qa"]
    doc_char_limit = cfg["corpus"]["doc_char_limit"]
    llm = build_llm(qa_cfg)

    chunks = get_chunks(config=cfg)
    max_chunks = qa_cfg.get("max_chunks")
    if max_chunks:
        chunks = chunks[:max_chunks]
    questions_per_chunk = qa_cfg.get("questions_per_chunk") or 1

    bank = load_existing_bank()
    seen_chunk_ids = {item["chunk_id"] for item in bank}

    QA_BANK_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = len(chunks) * questions_per_chunk
    generated = len(bank)
    for chunk in chunks:
        if chunk["chunk_id"] in seen_chunk_ids:
            continue
        chunk_text = chunk["text"][:doc_char_limit]
        if looks_like_index(chunk_text):
            print(f"Skipping chunk {chunk['chunk_id']}: looks like an index or table of contents")
            continue
        for _ in range(questions_per_chunk):
            item = generate_one_qa(llm, chunk_text)
            if item is None:
                print(f"Skipping chunk {chunk['chunk_id']}: failed validation twice")
                continue
            bank.append({
                "chunk_id": chunk["chunk_id"],
                "split": chunk["split"],
                "question": item["question"],
                "answer": item["answer"],
                "key_fact": item["key_fact"],
            })
            with open(QA_BANK_PATH, "w", encoding="utf-8") as f:
                json.dump(bank, f, indent=2, ensure_ascii=False)
            generated += 1
            if generated % 10 == 0 or generated == total:
                print(f"Generated {generated}/{total}")

    return bank


if __name__ == "__main__":
    generate_qa_bank()
