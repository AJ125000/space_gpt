from fastapi import FastAPI
from langchain_core.messages import HumanMessage, AIMessage

from .schemas import ChatRequest
from .graph import graph_app

# Initialize FastAPI app
api = FastAPI(
    title="Space-GPT API",
    description="API for the AI-powered space research assistant",
    version="1.0.0",
)

@api.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Receives a chat request and returns the chatbot's response.
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