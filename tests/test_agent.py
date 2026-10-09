import unittest
from src.state import AgentState
from src.agent import (
    planner_node,
    reformulate_query,
    should_continue,
    build_raft_graph,
    run_raft_agent,
)
from langgraph.graph import END


def test_agent_state_keys():
    expected_keys = {
        "query",
        "original_query",
        "attempt_count",
        "max_attempts",
        "retrieved_docs",
        "chain_of_thought",
        "status",
        "final_answer",
    }
    assert expected_keys.issubset(AgentState.__annotations__.keys())


def test_planner_reformulation():
    initial_state: AgentState = {
        "query": "brake caliper torque",
        "original_query": "brake caliper torque",
        "attempt_count": 0,
        "max_attempts": 3,
        "retrieved_docs": [],
        "chain_of_thought": None,
        "status": "initialized",
        "final_answer": "",
    }

    res1 = planner_node(initial_state)
    assert res1["attempt_count"] == 1
    assert res1["query"] == "brake caliper torque"

    state_retry = dict(initial_state)
    state_retry["attempt_count"] = 1
    res2 = planner_node(state_retry)
    assert res2["attempt_count"] == 2
    assert res2["query"] != "brake caliper torque"


def test_conditional_edge_routing():
    state_success: AgentState = {
        "query": "q",
        "original_query": "q",
        "attempt_count": 1,
        "max_attempts": 3,
        "retrieved_docs": [],
        "chain_of_thought": "Golden doc identified.",
        "status": "answer_found",
        "final_answer": "154 ft-lbf",
    }
    assert should_continue(state_success) == END

    state_retry: AgentState = {
        "query": "q",
        "original_query": "q",
        "attempt_count": 1,
        "max_attempts": 3,
        "retrieved_docs": [],
        "chain_of_thought": "All 4 docs are distractors.",
        "status": "all_distractors",
        "final_answer": "None",
    }
    assert should_continue(state_retry) == "planner"

    state_exhausted: AgentState = {
        "query": "q",
        "original_query": "q",
        "attempt_count": 3,
        "max_attempts": 3,
        "retrieved_docs": [],
        "chain_of_thought": "Still distractors.",
        "status": "all_distractors",
        "final_answer": "None",
    }
    assert should_continue(state_exhausted) == "fallback"


def test_end_to_end_immediate_success():
    def mock_retriever(query: str, k: int):
        return ["[Doc 0]: Lug nut torque is 154 ft-lbf for steel wheels."]

    def mock_synthesizer(query: str, docs: list):
        return {
            "chain_of_thought": "Doc 0 directly specifies 154 ft-lbf.",
            "status": "answer_found",
            "answer": "154 ft-lbf (209 N-m)",
        }

    final_state = run_raft_agent(
        query="What is the wheel nut torque?",
        max_attempts=3,
        retriever_func=mock_retriever,
        llm_invoker=mock_synthesizer,
    )

    assert final_state["attempt_count"] == 1
    assert final_state["status"] == "answer_found"
    assert "154 ft-lbf" in final_state["final_answer"]


def test_end_to_end_self_correcting_retry():
    history = []

    def mock_retriever(query: str, k: int):
        history.append(query)
        if len(history) == 1:
            return ["[Doc 0]: Oil filter capacity 0.5L."]
        return ["[Doc 0]: Brake caliper slide pin torque is 20 ft-lbf."]

    def mock_synthesizer(query: str, docs: list):
        if len(history) == 1:
            return {
                "chain_of_thought": "Doc 0 is engine oil, irrelevant to caliper torque.",
                "status": "all_distractors",
                "answer": "None",
            }
        return {
            "chain_of_thought": "Doc 0 provides the caliper slide pin torque.",
            "status": "answer_found",
            "answer": "20 ft-lbf (27 N-m)",
        }

    final_state = run_raft_agent(
        query="caliper torque",
        max_attempts=3,
        retriever_func=mock_retriever,
        llm_invoker=mock_synthesizer,
    )

    assert final_state["attempt_count"] == 2
    assert final_state["status"] == "answer_found"
    assert "20 ft-lbf" in final_state["final_answer"]
    assert len(history) == 2
    assert history[0] != history[1]


def test_end_to_end_max_retries_exceeded():
    def mock_retriever(query: str, k: int):
        return ["[Doc 0]: Irrelevant text."]

    def mock_synthesizer(query: str, docs: list):
        return {
            "chain_of_thought": "Distractor only.",
            "status": "all_distractors",
            "answer": "None",
        }

    final_state = run_raft_agent(
        query="nonexistent spec",
        max_attempts=3,
        retriever_func=mock_retriever,
        llm_invoker=mock_synthesizer,
    )

    assert final_state["attempt_count"] == 3
    assert final_state["status"] == "max_retries_exceeded"
    assert "Unable to find verified specifications" in final_state["final_answer"]


class TestWeek3Agent(unittest.TestCase):
    def test_agent_state_keys(self):
        test_agent_state_keys()

    def test_planner_reformulation(self):
        test_planner_reformulation()

    def test_conditional_edge_routing(self):
        test_conditional_edge_routing()

    def test_end_to_end_immediate_success(self):
        test_end_to_end_immediate_success()

    def test_end_to_end_self_correcting_retry(self):
        test_end_to_end_self_correcting_retry()

    def test_end_to_end_max_retries_exceeded(self):
        test_end_to_end_max_retries_exceeded()


if __name__ == "__main__":
    unittest.main(verbosity=2)
