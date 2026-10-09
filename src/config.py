import os
import copy
import yaml

DEFAULTS = {
    "corpus": {
        "pdf_dir": "data/raw_pdfs",
        "chroma_path": "data/chromadb",
        "collection": "raft_corpus",
        "embed_model": "nomic-embed-text",
        "chunk_size": 1800,
        "chunk_overlap": 200,
        "doc_char_limit": 450,
        "eval_split": 0.15,
    },
    "qa": {
        "backend": "ollama",
        "model": "llama3.2:3b-instruct-q4_K_M",
        "base_url": None,
        "questions_per_chunk": 1,
        "max_chunks": None,
    },
    "dataset": {
        "n_distractors": 3,
        "p_golden": 0.85,
        "p_unanswerable": 0.15,
        "hard_negative_ratio": 0.5,
        "seed": 42,
        "domain_label": "the provided documents",
    },
    "model": {
        "base_model": "unsloth/Llama-3.2-3B-Instruct",
        "max_seq_length": 1536,
        "load_in_4bit": True,
        "lora_r": 16,
        "lora_alpha": 16,
        "batch_size": 1,
        "grad_accum": 4,
        "lr": 2.0e-4,
        "epochs": 1,
    },
    "agent": {
        "serve_model": "raft-llama3.2",
        "max_attempts": 3,
        "k": 4,
        "temperature": 0.0,
    },
    "eval": {
        "judge_backend": "ollama",
        "judge_model": "llama3.2:3b-instruct-q4_K_M",
        "judge_base_url": None,
        "embed_model": "nomic-embed-text",
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
    },
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")


def _deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None):
    if path is None:
        path = _DEFAULT_CONFIG_PATH
        if not os.path.exists(path):
            return copy.deepcopy(DEFAULTS)
    with open(path, "r") as f:
        override = yaml.safe_load(f) or {}
    return _deep_merge(copy.deepcopy(DEFAULTS), override)


if __name__ == "__main__":
    cfg = load_config()
    assert cfg["corpus"]["chunk_size"] == 1800
    assert cfg["agent"]["k"] == 4

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("corpus:\n  chunk_size: 999\n")
        tmp_path = f.name
    try:
        overridden = load_config(tmp_path)
        assert overridden["corpus"]["chunk_size"] == 999
        assert overridden["corpus"]["embed_model"] == "nomic-embed-text"
        assert overridden["model"]["base_model"] == "unsloth/Llama-3.2-3B-Instruct"
    finally:
        os.remove(tmp_path)

    print("config.py self-check passed")
