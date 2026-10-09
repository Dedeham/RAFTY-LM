import json
import logging
from typing import Callable, Optional, Dict, Any, List
from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.config import load_config
from src.train import build_system_prompt
from src.jsonutil import strip_json_fences

logger = logging.getLogger("agent")

_cfg = load_config()
DEFAULT_MODEL_NAME = _cfg["agent"]["serve_model"]
DEFAULT_MAX_ATTEMPTS = _cfg["agent"]["max_attempts"]
DEFAULT_RETRIEVAL_K = _cfg["agent"]["k"]
DEFAULT_TEMPERATURE = _cfg["agent"]["temperature"]
DEFAULT_DOC_CHAR_LIMIT = _cfg["corpus"]["doc_char_limit"]
DEFAULT_DOMAIN_LABEL = _cfg["dataset"]["domain_label"]


def reformulate_query(query: str, attempt: int) -> str:
    strategies = [
        lambda q: q,
        lambda q: f"{q} specification detail",
        lambda q: f"Please provide detailed information regarding: {q}",
        lambda q: f"{q} full description overview",
    ]
    strategy = strategies[min(attempt, len(strategies) - 1)]
    return strategy(query)


def planner_node(state: AgentState) -> Dict[str, Any]:
    current_attempt = state.get("attempt_count", 0)
    original_query = state.get("original_query") or state.get("query", "")

    if current_attempt == 0:
        new_query = original_query
    else:
        new_query = reformulate_query(original_query, current_attempt)

    return {
        "query": new_query,
        "original_query": original_query,
        "attempt_count": current_attempt + 1,
    }


def make_retriever_node(
    retriever_func: Optional[Callable[[str, int], List[str]]] = None,
    k: int = DEFAULT_RETRIEVAL_K,
    doc_char_limit: int = DEFAULT_DOC_CHAR_LIMIT,
):
    def default_retrieve(query: str, k: int = k) -> List[str]:
        from src.store import get_retriever, format_doc, normalize_whitespace

        retriever = get_retriever(k=k)
        docs = retriever.invoke(query)
        formatted = []
        for i, doc in enumerate(docs):
            meta = doc.metadata or {}
            source_file = meta.get("source_file", "unknown")
            page = meta.get("page_number", "?")
            text = normalize_whitespace(doc.page_content)[:doc_char_limit]
            formatted.append(format_doc(i, source_file, page, text))
        return formatted

    fn = retriever_func or default_retrieve

    def retriever_node(state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        docs = fn(query, k)
        return {"retrieved_docs": docs}

    return retriever_node


def make_synthesizer_node(
    llm_invoker: Optional[Callable[[str, List[str]], Dict[str, Any]]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    domain_label: str = DEFAULT_DOMAIN_LABEL,
):
    def default_invoke_llm(query: str, docs: List[str]) -> Dict[str, Any]:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOllama(model=model_name, temperature=temperature)
        context_str = "\n\n".join(docs)
        user_prompt = f"Question: {query}\n\nContext Documents:\n{context_str}"
        messages = [
            SystemMessage(content=build_system_prompt(domain_label)),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        content = response.content.strip()

        try:
            parsed = json.loads(content)
        except Exception:
            parsed = json.loads(strip_json_fences(content))

        return parsed

    fn = llm_invoker or default_invoke_llm

    def synthesizer_node(state: AgentState) -> Dict[str, Any]:
        query = state["original_query"]
        docs = state.get("retrieved_docs", [])

        try:
            result = fn(query, docs)
            chain_of_thought = result.get("chain_of_thought", "")
            status = result.get("status", "all_distractors")
            answer = result.get("answer", "")
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            chain_of_thought = f"LLM synthesis failed with error: {str(e)}"
            status = "all_distractors"
            answer = "Could not synthesize answer due to model error."

        return {
            "chain_of_thought": chain_of_thought,
            "status": status,
            "final_answer": answer,
        }

    return synthesizer_node


def fallback_node(state: AgentState) -> Dict[str, Any]:
    attempts = state.get("attempt_count", 0)
    answer = state.get("final_answer")
    if not answer or answer == "None":
        answer = (
            f"Unable to find verified specifications after {attempts} attempts. "
            f"All candidate documents were classified as distractors."
        )
    return {
        "status": "max_retries_exceeded",
        "final_answer": answer,
    }


def should_continue(state: AgentState) -> str:
    status = state.get("status")
    attempts = state.get("attempt_count", 0)
    max_attempts = state.get("max_attempts", DEFAULT_MAX_ATTEMPTS)

    if status == "answer_found":
        return END

    if attempts < max_attempts:
        return "planner"

    return "fallback"


def build_raft_graph(
    retriever_func: Optional[Callable[[str, int], List[str]]] = None,
    llm_invoker: Optional[Callable[[str, List[str]], Dict[str, Any]]] = None,
    k: int = DEFAULT_RETRIEVAL_K,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", make_retriever_node(retriever_func, k=k))
    graph.add_node("synthesizer", make_synthesizer_node(llm_invoker, model_name=model_name, temperature=temperature))
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("fallback", END)

    graph.add_conditional_edges(
        "synthesizer",
        should_continue,
        {
            "planner": "planner",
            "fallback": "fallback",
            END: END,
        },
    )

    return graph.compile()


def run_raft_agent(
    query: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retriever_func: Optional[Callable[[str, int], List[str]]] = None,
    llm_invoker: Optional[Callable[[str, List[str]], Dict[str, Any]]] = None,
    k: int = DEFAULT_RETRIEVAL_K,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
) -> AgentState:
    app = build_raft_graph(
        retriever_func=retriever_func,
        llm_invoker=llm_invoker,
        k=k,
        model_name=model_name,
        temperature=temperature,
    )
    initial_state: AgentState = {
        "query": query,
        "original_query": query,
        "attempt_count": 0,
        "max_attempts": max_attempts,
        "retrieved_docs": [],
        "chain_of_thought": None,
        "status": "initialized",
        "final_answer": "",
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    import sys

    test_query = sys.argv[1] if len(sys.argv) > 1 else "What is the brake caliper slide pin torque?"
    print(f"Running RAFT Agent on query: {test_query}\n")
    final_state = run_raft_agent(test_query)
    print("Execution Finished!")
    print(f"Status: {final_state['status']}")
    print(f"Attempts: {final_state['attempt_count']}")
    print(f"Final Query Used: {final_state['query']}")
    print(f"Chain of Thought: {final_state['chain_of_thought']}")
    print(f"Final Answer: {final_state['final_answer']}")
