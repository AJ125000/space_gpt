from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# --- API Schemas ---
class ChatRequest(BaseModel):
    query: str
    chat_history: List[dict] = Field(
        default_factory=list, 
        description="A list of previous messages, e.g., [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello'}]"
    )

# --- Graph State Schema ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    original_query: str
    chat_history: List[BaseMessage]
    rag_query: str
    search_query: str
    is_out_of_scope: bool
    retrieved_docs: str  # Changed to string to accept the formatted output from KnowledgeBase
    search_results: str
    filtered_context: str
    final_answer: str