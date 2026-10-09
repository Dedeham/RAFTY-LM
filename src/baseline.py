import json

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from src.store import get_retriever, format_doc, normalize_whitespace
from src.train import build_system_prompt
from src.config import load_config
from src.jsonutil import strip_json_fences

_cfg = load_config()
DEFAULT_MODEL_NAME = _cfg["agent"]["serve_model"]
DEFAULT_TEMPERATURE = _cfg["agent"]["temperature"]
DEFAULT_RETRIEVAL_K = _cfg["agent"]["k"]
DEFAULT_DOC_CHAR_LIMIT = _cfg["corpus"]["doc_char_limit"]
DEFAULT_DOMAIN_LABEL = _cfg["dataset"]["domain_label"]


class NaiveRAG:
    def __init__(
        self,
        k=DEFAULT_RETRIEVAL_K,
        model_name=DEFAULT_MODEL_NAME,
        temperature=DEFAULT_TEMPERATURE,
        domain_label=DEFAULT_DOMAIN_LABEL,
        doc_char_limit=DEFAULT_DOC_CHAR_LIMIT,
    ):
        self.retriever = get_retriever(k=k)
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.system_prompt = build_system_prompt(domain_label)
        self.doc_char_limit = doc_char_limit

    def query(self, question):
        retrieved_docs = self.retriever.invoke(question)

        formatted_docs = []
        normalized_texts = []
        for i, doc in enumerate(retrieved_docs):
            meta = doc.metadata or {}
            source_file = meta.get("source_file", "unknown")
            page = meta.get("page_number", "?")
            text = normalize_whitespace(doc.page_content)[: self.doc_char_limit]
            normalized_texts.append(text)
            formatted_docs.append(format_doc(i, source_file, page, text))

        context_str = "\n\n".join(formatted_docs)
        user_prompt = f"Question: {question}\n\nContext Documents:\n{context_str}"
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]

        response = self.llm.invoke(messages)
        content = response.content.strip()

        try:
            parsed = json.loads(content)
        except Exception:
            try:
                parsed = json.loads(strip_json_fences(content))
            except Exception:
                parsed = {"chain_of_thought": "", "status": "unparsed", "answer": content}

        return {
            "question": question,
            "response": parsed.get("answer", ""),
            "status": parsed.get("status", "unparsed"),
            "chain_of_thought": parsed.get("chain_of_thought", ""),
            "retrieved_chunks": [
                {
                    "source": doc.metadata.get("source_file"),
                    "page": doc.metadata.get("page_number"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "content_snippet": text,
                }
                for doc, text in zip(retrieved_docs, normalized_texts)
            ],
        }


if __name__ == "__main__":
    rag = NaiveRAG()
    test_q = "What is the wheel nut torque specification?"
    print(f"Question: {test_q}\n")
    res = rag.query(test_q)
    print("Answer:\n", res["response"])
