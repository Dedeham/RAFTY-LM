.PHONY: ingest qa dataset train eval agent test clean

ingest:
	python3 -m src.store

qa:
	python3 -m src.qa_gen

dataset:
	python3 -m src.dataset

train:
	python3 -m src.train

eval:
	python3 -m src.evaluate

agent:
	python3 -m src.agent

test:
	python3 -m pytest tests/ -v

clean:
	rm -rf data/chromadb data/qa_bank.json data/raft_train.json data/raft_eval.json eval/ __pycache__ src/__pycache__ tests/__pycache__ unsloth_compiled_cache
