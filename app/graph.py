from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import os

from .schemas import GraphState
from core.retriever import knowledge_base
from core.tools import web_search_tool
from .config import settings

os.environ["GOOGLE_API_KEY"] = settings.google_api_key
# Initialize the Gemini LLM for the graph nodes
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    api_key=settings.google_api_key
)

# --- NODE DEFINITIONS ---

class Plan(BaseModel):
    rag_query: str = Field(description="A precise, keyword-rich query for vector search.")
    search_query: str = Field(description="A concise query for a web search engine.")
    is_out_of_scope: bool = Field(description="True if the query is NOT related to space, astronomy, or astrophysics.")

def plan_node(state: GraphState):
    print("---PLANNING---")
    prompt = ChatPromptTemplate.from_template(
        """You are a query understanding engine. Analyze the user's query and chat history.
         If the query is NOT related to space, ISRO, NASA, rockets, astronomy, or astrophysics, set `is_out_of_scope` to True.
         Otherwise, generate a `rag_query` for vector retrieval and a `search_query` for web search.
         Respond with a JSON object.

         Chat History: {chat_history}
         User Query: {query}"""
    )
    chain = prompt | llm.with_structured_output(Plan)
    
    # Format chat history from messages
    chat_history_str = ""
    if state.get("messages"):
        chat_history_str = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
            for msg in state["messages"][:-1]  # Exclude the current message
        ])
    
    result = chain.invoke({
        "chat_history": chat_history_str, 
        "query": state["original_query"]
    })
    return {"rag_query": result.rag_query, "search_query": result.search_query, "is_out_of_scope": result.is_out_of_scope}

async def retrieve_and_search_node(state: GraphState):
    print("---RETRIEVING & SEARCHING (PARALLEL)---")
    rag_query = state["rag_query"]
    search_query = state["search_query"]

    # Run knowledge base retrieval and web search concurrently
    results = await asyncio.gather(
        knowledge_base.retrieve(rag_query),
        asyncio.to_thread(web_search_tool.run, search_query)
    )
    return {"retrieved_docs": results[0], "search_results": results[1]}

def critique_node(state: GraphState):
    print("---CRITIQUING & FILTERING---")
    prompt = ChatPromptTemplate.from_template(
        """You are a relevance analysis agent. Examine the retrieved documents and web search results against the original user query.
         Filter out any information that is not directly relevant.
         Synthesize the remaining pieces into a single, consolidated context.

         Original Query: {original_query}
         Knowledge Base Docs: {retrieved_docs}
         Web Search Results: {search_results}"""
    )
    chain = prompt | llm
    result = chain.invoke({
        "original_query": state["original_query"],
        "retrieved_docs": state["retrieved_docs"],
        "search_results": state["search_results"],
    })
    return {"filtered_context": result.content}

def writer_node(state: GraphState):
    print("---WRITING FINAL ANSWER---")
    prompt = ChatPromptTemplate.from_template(
        """You are a final answer synthesizer. Craft a comprehensive, well-structured answer using the provided 'Filtered Context'.
         If the context is empty, inform the user you couldn't find relevant information.
         Include chat history context if relevant to provide a conversational flow.

         Chat History: {chat_history}
         User's Original Query: {original_query}
         Filtered Context: {filtered_context}"""
    )
    chain = prompt | llm
    
    # Format chat history for the prompt
    chat_history_str = ""
    if state.get("messages"):
        chat_history_str = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
            for msg in state["messages"][-5:]  # Include last 5 messages for context
        ])
    
    result = chain.invoke({
        "original_query": state["original_query"], 
        "filtered_context": state["filtered_context"],
        "chat_history": chat_history_str
    })
    return {"final_answer": result.content}


def out_of_scope_node(state: GraphState):
    print("---HANDLING OUT OF SCOPE---")
    final_answer = "I'm sorry, but I am a specialized chatbot for space-related topics. I can't help with that."
    print(f"Final Answer: {final_answer}")
    return {"final_answer": final_answer}

# --- GRAPH ASSEMBLY ---
def create_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("planner", plan_node)
    workflow.add_node("retrieve_and_search", retrieve_and_search_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    
    workflow.set_entry_point("planner")

    def should_continue(state: GraphState):
        """
        Conditional edge function to decide the next node based on 'is_out_of_scope'.
        """
        if state["is_out_of_scope"]:
            return "out_of_scope"
        else:
            return "retrieve_and_search"

    workflow.add_conditional_edges(
        "planner",
        should_continue,
        {
            "retrieve_and_search": "retrieve_and_search",
            "out_of_scope": "out_of_scope"
        }
    )
    workflow.add_edge("retrieve_and_search", "critique")
    workflow.add_edge("critique", "writer")
    workflow.add_edge("writer", END)
    workflow.add_edge("out_of_scope", END)
    
    # Create memory saver for checkpointing (enables streaming with state persistence)
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

# Create both streaming and non-streaming versions
graph_app = create_graph()

# Function to handle streaming with LangGraph
def create_streaming_graph():
    """Create a graph optimized for streaming responses"""
    return create_graph()

streaming_graph_app = create_streaming_graph()