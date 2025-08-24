from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
import json
import uuid
import os

from .schemas import ChatRequest
from .graph import graph_app
from .config import settings

# Initialize FastAPI app
api = FastAPI(
    title="Space-GPT API",
    description="API for the AI-powered space research assistant",
    version="1.0.0",
)

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set LangSmith environment variables
os.environ["LANGSMITH_TRACING_V2"] = settings.langsmith_tracing_v2
os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

# Global dictionary to store step updates
step_updates = {}

async def execute_graph_with_steps(inputs, session_id):
    """Execute the graph and yield step updates"""
    
    # Step 1: Planning
    yield {"type": "step", "step": "Planning query analysis...", "session_id": session_id}
    
    # Get initial planning
    planning_state = {
        "original_query": inputs["original_query"],
        "chat_history": inputs["chat_history"],
        "rag_query": "",
        "search_query": "",
        "is_out_of_scope": False,
        "retrieved_docs": "",
        "search_results": "",
        "filtered_context": "",
        "final_answer": "",
    }
    
    # Import here to avoid circular imports
    from .graph import plan_node, retrieve_and_search_node, critique_node, writer_node, out_of_scope_node
    
    # Execute planning
    plan_result = plan_node(planning_state)
    planning_state.update(plan_result)
    
    # Check if out of scope
    if planning_state["is_out_of_scope"]:
        yield {"type": "step", "step": "Handling out-of-scope query...", "session_id": session_id}
        final_result = out_of_scope_node(planning_state)
        planning_state.update(final_result)
        yield {"type": "answer", "answer": planning_state["final_answer"], "session_id": session_id}
        return
    
    # Step 2: Retrieval and Search
    yield {"type": "step", "step": "Retrieving from knowledge base and searching...", "session_id": session_id}
    retrieve_result = await retrieve_and_search_node(planning_state)
    planning_state.update(retrieve_result)
    
    # Step 3: Critique and Filter
    yield {"type": "step", "step": "Filtering and analyzing context...", "session_id": session_id}
    critique_result = critique_node(planning_state)
    planning_state.update(critique_result)
    
    # Step 4: Generate Final Answer
    yield {"type": "step", "step": "Generating final response...", "session_id": session_id}
    writer_result = writer_node(planning_state)
    planning_state.update(writer_result)
    
    # Send final answer
    yield {"type": "answer", "answer": planning_state["final_answer"], "session_id": session_id}

@api.post("/chat-stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Receives a chat request and returns streaming updates of the processing steps.
    """
    session_id = str(uuid.uuid4())

    async def generate_stream():
        try:
            # Convert chat history from dicts to BaseMessage objects
            chat_history = []
            for msg in request.chat_history:
                if msg.get("role") == "user":
                    chat_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") in ("assistant", "ai"):
                    chat_history.append(AIMessage(content=msg.get("content", "")))

            inputs = {
                "original_query": request.query,
                "chat_history": chat_history,
            }
            
            # Send initial step
            yield f"data: {json.dumps({'type': 'step', 'step': 'Initializing...', 'session_id': session_id})}\n\n"
            
            # Execute graph with step updates
            async for update in execute_graph_with_steps(inputs, session_id):
                yield f"data: {json.dumps(update)}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
            
        except Exception as e:
            print(f"Error in chat stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'session_id': session_id})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Receives a chat request and returns the chatbot's response (non-streaming version).
    """
    # Convert chat history from dicts to BaseMessage objects
    chat_history = []
    for msg in request.chat_history:
        if msg.get("role") == "user":
            chat_history.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") in ("assistant", "ai"):
            chat_history.append(AIMessage(content=msg.get("content", "")))

    inputs = {
        "original_query": request.query,
        "chat_history": chat_history,
    }
    
    # Asynchronously invoke the LangGraph agent
    final_state = await graph_app.ainvoke(inputs)
    
    return {"answer": final_state.get("final_answer", "Sorry, something went wrong.")}

@api.get("/")
def read_root():
    return {"message": "Welcome to the Space-GPT API. Go to /docs for the API documentation."}