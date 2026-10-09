import json
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

from src.config import load_config
from src.baseline import NaiveRAG
from src.agent import run_raft_agent

EVAL_SET_PATH = Path("data/raft_eval.json")
OUTPUT_DIR = Path("eval")
RAW_OUTPUT_PATH = OUTPUT_DIR / "raw.json"
REPORT_PATH = OUTPUT_DIR / "report.md"

GROUND_TRUTH_ABSTAIN_STATUS = "all_distractors"
AGENT_ABSTAIN_STATUSES = {"all_distractors", "max_retries_exceeded"}
BASELINE_ABSTAIN_STATUSES = {"all_distractors"}
NOT_ANSWERABLE_PHRASE = "do not contain the necessary information"
SYNTHESIS_FAILURE_PREFIX = "LLM synthesis failed"
RAGAS_METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def answer_recalled(predicted, ground_truth):
    predicted_n = normalize_text(predicted)
    truth_n = normalize_text(ground_truth)
    if not truth_n or not predicted_n:
        return False
    if truth_n in predicted_n:
        return True
    words = [w for w in truth_n.split() if len(w) > 3]
    if not words:
        return False
    hits = sum(1 for w in words if w in predicted_n)
    return hits / len(words) >= 0.6


def load_eval_set(path=EVAL_SET_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Eval set not found at {path}. Run src/dataset.py first to build it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_baseline_item(rag, item):
    start = time.time()
    result = rag.query(item["question"])
    elapsed = time.time() - start
    contexts = [c.get("content_snippet", "") for c in result.get("retrieved_chunks", [])]
    parse_ok = result.get("status") != "unparsed"
    return {
        "question": item["question"],
        "predicted_answer": result.get("response", ""),
        "predicted_status": result.get("status", "unparsed"),
        "chain_of_thought": result.get("chain_of_thought", ""),
        "contexts": contexts,
        "latency_seconds": round(elapsed, 3),
        "json_parsed": parse_ok,
    }


def run_agent_item(item, agent_kwargs):
    start = time.time()
    state = run_raft_agent(item["question"], **agent_kwargs)
    elapsed = time.time() - start
    contexts = state.get("retrieved_docs", [])
    cot = state.get("chain_of_thought") or ""
    parse_ok = not cot.startswith(SYNTHESIS_FAILURE_PREFIX)
    return {
        "question": item["question"],
        "predicted_answer": state.get("final_answer", ""),
        "predicted_status": state.get("status", "max_retries_exceeded"),
        "chain_of_thought": cot,
        "contexts": contexts,
        "latency_seconds": round(elapsed, 3),
        "json_parsed": parse_ok,
        "attempts": state.get("attempt_count", 0),
    }


def is_predicted_abstain(predicted_status, predicted_answer, abstain_statuses):
    if predicted_status in abstain_statuses:
        return True
    return NOT_ANSWERABLE_PHRASE in normalize_text(predicted_answer)


def compute_deterministic_metrics(items, predictions, abstain_statuses):
    ground_truth_abstain = [i for i, item in enumerate(items) if item["status"] == GROUND_TRUTH_ABSTAIN_STATUS]
    ground_truth_answerable = [i for i, item in enumerate(items) if item["status"] != GROUND_TRUTH_ABSTAIN_STATUS]

    abstain_hits = sum(
        1 for i in ground_truth_abstain
        if is_predicted_abstain(predictions[i]["predicted_status"], predictions[i]["predicted_answer"], abstain_statuses)
    )
    abstention_recall = abstain_hits / len(ground_truth_abstain) if ground_truth_abstain else None

    answer_hits = sum(
        1 for i in ground_truth_answerable
        if answer_recalled(predictions[i]["predicted_answer"], items[i]["answer"])
    )
    answer_recall = answer_hits / len(ground_truth_answerable) if ground_truth_answerable else None

    json_parse_rate = sum(1 for p in predictions if p["json_parsed"]) / len(predictions) if predictions else None

    return {
        "abstention_recall": abstention_recall,
        "answer_recall": answer_recall,
        "json_parse_rate": json_parse_rate,
        "n_ground_truth_abstain": len(ground_truth_abstain),
        "n_ground_truth_answerable": len(ground_truth_answerable),
    }


def build_judge_llm(eval_cfg):
    backend = eval_cfg["judge_backend"]
    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=eval_cfg["judge_model"], base_url=eval_cfg.get("judge_base_url"))
    if backend == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for the openai judge backend")
        return ChatOpenAI(model=eval_cfg["judge_model"], base_url=eval_cfg.get("judge_base_url"), api_key=api_key)
    raise ValueError(f"Unsupported judge backend: {backend}")


def build_judge_embeddings(eval_cfg):
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=eval_cfg["embed_model"])


def run_ragas_eval(items, predictions, eval_cfg, metric_names):
    try:
        import sys
        import types
        import importlib

        try:
            importlib.import_module("langchain_community.chat_models.vertexai")
        except ImportError:
            stub = types.ModuleType("langchain_community.chat_models.vertexai")
            stub.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules["langchain_community.chat_models.vertexai"] = stub

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        print(f"Skipping Ragas metrics: package not available ({e}). Install with `pip install ragas`.")
        return None

    metric_lookup = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    metrics = [metric_lookup[name] for name in metric_names if name in metric_lookup]

    try:
        dataset = Dataset.from_dict({
            "question": [item["question"] for item in items],
            "answer": [p["predicted_answer"] for p in predictions],
            "contexts": [p["contexts"] or [""] for p in predictions],
            "ground_truth": [item["answer"] for item in items],
        })

        judge_llm = LangchainLLMWrapper(build_judge_llm(eval_cfg))
        judge_embeddings = LangchainEmbeddingsWrapper(build_judge_embeddings(eval_cfg))

        result = evaluate(dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)
        scores_df = result.to_pandas()

        per_item_scores = []
        for _, row in scores_df.iterrows():
            per_item_scores.append({name: row.get(name) for name in metric_names if name in row})

        averages = {name: float(scores_df[name].mean()) for name in metric_names if name in scores_df.columns}
        return {"averages": averages, "per_item": per_item_scores}
    except Exception as e:
        print(f"Skipping Ragas metrics: evaluation failed ({e}).")
        return None


def format_pct(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def format_score(value):
    return "n/a" if value is None else f"{value:.3f}"


def write_report(items, baseline_metrics, agent_metrics, baseline_ragas, agent_ragas, eval_cfg, metric_names):
    lines = [
        "# RAFT Agent vs Naive Baseline Evaluation",
        "",
        f"Eval set size: {len(items)}",
        f"Judge backend: {eval_cfg['judge_backend']} ({eval_cfg['judge_model']})",
        "",
        "| Metric | Baseline | RAFT Agent |",
        "|---|---|---|",
        f"| Abstention recall | {format_pct(baseline_metrics['abstention_recall'])} | {format_pct(agent_metrics['abstention_recall'])} |",
        f"| Answer recall | {format_pct(baseline_metrics['answer_recall'])} | {format_pct(agent_metrics['answer_recall'])} |",
        f"| JSON parse rate | {format_pct(baseline_metrics['json_parse_rate'])} | {format_pct(agent_metrics['json_parse_rate'])} |",
    ]

    if baseline_ragas or agent_ragas:
        for name in metric_names:
            b_val = baseline_ragas["averages"].get(name) if baseline_ragas else None
            a_val = agent_ragas["averages"].get(name) if agent_ragas else None
            lines.append(f"| {name} (Ragas) | {format_score(b_val)} | {format_score(a_val)} |")
    else:
        lines.append("")
        lines.append("Ragas metrics were skipped (package not installed).")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_evaluation():
    cfg = load_config()
    items = load_eval_set()

    agent_cfg = cfg["agent"]
    dataset_cfg = cfg["dataset"]
    corpus_cfg = cfg["corpus"]

    print(f"Loaded {len(items)} eval items from {EVAL_SET_PATH}")

    rag = NaiveRAG(
        k=agent_cfg["k"],
        model_name=agent_cfg["serve_model"],
        temperature=agent_cfg["temperature"],
        domain_label=dataset_cfg["domain_label"],
        doc_char_limit=corpus_cfg["doc_char_limit"],
    )
    agent_kwargs = {
        "max_attempts": agent_cfg["max_attempts"],
        "k": agent_cfg["k"],
        "model_name": agent_cfg["serve_model"],
        "temperature": agent_cfg["temperature"],
    }

    baseline_predictions = []
    agent_predictions = []
    print("Running baseline (NaiveRAG)...")
    for item in tqdm(items):
        baseline_predictions.append(run_baseline_item(rag, item))

    print("Running RAFT agent...")
    for item in tqdm(items):
        agent_predictions.append(run_agent_item(item, agent_kwargs))

    baseline_metrics = compute_deterministic_metrics(items, baseline_predictions, BASELINE_ABSTAIN_STATUSES)
    agent_metrics = compute_deterministic_metrics(items, agent_predictions, AGENT_ABSTAIN_STATUSES)

    metric_names = cfg["eval"].get("metrics") or RAGAS_METRIC_NAMES

    print("Running Ragas metrics (baseline)...")
    baseline_ragas = run_ragas_eval(items, baseline_predictions, cfg["eval"], metric_names)
    print("Running Ragas metrics (agent)...")
    agent_ragas = run_ragas_eval(items, agent_predictions, cfg["eval"], metric_names)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_output = {
        "eval_set_size": len(items),
        "judge_model": cfg["eval"]["judge_model"],
        "baseline": {
            "predictions": baseline_predictions,
            "metrics": baseline_metrics,
            "ragas": baseline_ragas,
        },
        "agent": {
            "predictions": agent_predictions,
            "metrics": agent_metrics,
            "ragas": agent_ragas,
        },
    }
    with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_output, f, indent=2, ensure_ascii=False)

    write_report(items, baseline_metrics, agent_metrics, baseline_ragas, agent_ragas, cfg["eval"], metric_names)

    print(f"Wrote {RAW_OUTPUT_PATH} and {REPORT_PATH}")
    return raw_output


if __name__ == "__main__":
    run_evaluation()
