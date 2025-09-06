from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

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
    messages: Annotated[List[BaseMessage], add_messages]  # Chat history with add reduction
    original_query: str
    rag_query: str
    search_query: str
    is_out_of_scope: bool
    retrieved_docs: str  # Changed to string to accept the formatted output from KnowledgeBase
    search_results: str
    filtered_context: str
    final_answer: str