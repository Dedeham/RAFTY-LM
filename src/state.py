from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    query: str
    original_query: str
    attempt_count: int
    max_attempts: int
    retrieved_docs: List[str]
    chain_of_thought: Optional[str]
    status: str
    final_answer: str
