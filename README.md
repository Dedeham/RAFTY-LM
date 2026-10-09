# RAFTY-LM

A domain-agnostic RAFT (Retrieval-Augmented Fine-Tuning) pipeline. Bring your own PDFs, fine-tune a local LLM with Unsloth QLoRA to reject distractor documents and reduce RAG hallucination, and evaluate with Ragas plus deterministic metrics. The model learns to answer strictly from retrieved context and to abstain when the context doesn't contain the answer — that abstention behavior is the core RAFT training target, and what evaluation measures.

## Requirements

- NVIDIA GPU, 8GB+ VRAM (for training; ingest/qa/eval can run on CPU Ollama, just slower).
- Python 3.10+.
- [Ollama](https://ollama.com) running locally, with: `ollama pull llama3.2:3b-instruct-q4_K_M` and `ollama pull nomic-embed-text`.

## Quickstart

```
pip install -r requirements.txt
cp /path/to/your/*.pdf data/raw_pdfs/   # 1. add PDFs
make ingest                             # 2. chunk, embed, persist to ChromaDB
make qa                                 # 3. generate a QA bank with a local LLM
make dataset                            # 4. build RAFT train/eval sets
make train                              # 5. QLoRA fine-tune, export to GGUF

# 6. make train prints the exact `ollama create <name> -f <path>/Modelfile`
#    command to run here (path depends on the base model), then:
make eval
```

`make test` runs the unit suite; `make agent` queries the trained model interactively via the self-correcting agent.

## Architecture

| Stage | File | What it does |
|---|---|---|
| Ingest | `src/ingest.py` | Reads PDFs (PyMuPDF), chunks pages, hashes each chunk to a stable train/eval split. |
| Vector store | `src/store.py` | Embeds chunks via Ollama, persists to ChromaDB; `make ingest` runs this. Exposes the retriever. |
| QA generation | `src/qa_gen.py` | Prompts a local LLM for a grounded Q/A/key-fact per chunk; writes `data/qa_bank.json`. |
| Dataset build | `src/dataset.py` | Builds context windows (golden + hard/random distractors), forces some fully unanswerable. |
| Fine-tuning | `src/train.py` | 4-bit Unsloth load + QLoRA adapter, trains on RAFT examples, exports merged GGUF. |
| Agent | `src/agent.py` | LangGraph loop that re-retrieves on failure, up to `agent.max_attempts` times. |
| Baseline | `src/baseline.py` | Single-shot retrieve-then-generate RAG (`NaiveRAG`), the eval comparison point. |
| Evaluation | `src/evaluate.py` | Runs both over the eval set, computes deterministic + Ragas metrics, writes `eval/report.md`. |

## Config reference

Driven by `config.yaml` (schema in `src/config.py`, deep-merged over defaults):

| Group | Keys | Controls |
|---|---|---|
| `corpus` | `pdf_dir`, `chroma_path`, `collection`, `embed_model`, `chunk_size`, `chunk_overlap`, `doc_char_limit`, `eval_split` | PDF/store location, embedding model, chunking, train/eval split. |
| `qa` | `backend`, `model`, `base_url`, `questions_per_chunk`, `max_chunks` | QA-generation LLM, questions per chunk, chunk cap. |
| `dataset` | `n_distractors`, `p_golden`, `p_unanswerable`, `hard_negative_ratio`, `seed`, `domain_label` | Distractor mix, unanswerable fraction, seed, prompt domain phrase. |
| `model` | `base_model`, `max_seq_length`, `load_in_4bit`, `lora_r`, `lora_alpha`, `batch_size`, `grad_accum`, `lr`, `epochs` | Base model and QLoRA/training hyperparameters. |
| `agent` | `serve_model`, `max_attempts`, `k`, `temperature` | Inference model, retry budget, retrieval depth, temperature. |
| `eval` | `judge_backend`, `judge_model`, `judge_base_url`, `embed_model`, `metrics` | Ragas judge LLM/embeddings and metrics to compute. |

## Bring your own PDFs

1. Drop PDFs into `data/raw_pdfs/`.
2. Set `dataset.domain_label` in `config.yaml` to your subject area (used verbatim in the system prompt).
3. Re-run from `make ingest` onward.

**Copyright:** don't commit PDFs you can't redistribute — `data/raw_pdfs/*.pdf` is gitignored for this reason.

## Model options

Swap models with one config change, `model.base_model` in `config.yaml`:

| Model | VRAM | Notes |
|---|---|---|
| `unsloth/Llama-3.2-3B-Instruct` | ~8 GB | Default |
| `unsloth/Qwen2.5-1.5B-Instruct` | ~6 GB | Smallest option |
| `unsloth/Qwen2.5-7B-Instruct` | ~12 GB | Stronger reasoning |
| `unsloth/Mistral-7B-Instruct-v0.3` | ~12 GB | Alternative |
| `unsloth/Meta-Llama-3.1-8B-Instruct` | ~16 GB | Best results |

## Known limitations

- **1 epoch doesn't teach abstention.** Across two full runs on a 715-page automotive manual (964 chunks), the fine-tuned model never predicted `all_distractors` on ~30 held-out unanswerable questions per run (0.0% abstention recall, both runs) — it fabricates an answer instead. Answer recall/faithfulness were roughly on par with baseline. Verified twice, including on an unrelated finance PDF — this is a training-strength finding, not a pipeline bug. Try more `model.epochs`, a higher `dataset.p_unanswerable`, or more source data.
- **`context_precision` returns `nan`** with the default local judge (`llama3.2:3b-instruct-q4_K_M`) — it failed to parse a judgment for every item, in both runs. The other three Ragas metrics work fine; use a stronger `eval.judge_model` if you need this one.
- **Front-matter noise:** `qa_gen.py` doesn't filter title/copyright pages, so a few low-quality QA pairs about book metadata can appear — negligible on any reasonably sized corpus. Also, `data/qa_bank.json` and `data/raft_*.json` paths are relative to the working directory, so run pipeline commands from the repo root, as the Makefile does.

## License

Unlicense (public domain). See [LICENSE](LICENSE).
