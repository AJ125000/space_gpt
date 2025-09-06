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
        "messages": inputs["messages"],
        "original_query": inputs["original_query"],
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
            messages = []
            for msg in request.chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") in ("assistant", "ai"):
                    messages.append(AIMessage(content=msg.get("content", "")))
            
            # Add the current user message
            messages.append(HumanMessage(content=request.query))

            inputs = {
                "messages": messages,
                "original_query": request.query,
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

@api.post("/chat-stream-tokens")
async def chat_stream_tokens_endpoint(request: ChatRequest):
    """
    Token-by-token streaming endpoint for ChatGPT-style response generation.
    This streams individual tokens as they are generated by the LLM.
    """
    session_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": session_id}}

    async def generate_token_stream():
        try:
            # Convert chat history from dicts to BaseMessage objects
            messages = []
            for msg in request.chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") in ("assistant", "ai"):
                    messages.append(AIMessage(content=msg.get("content", "")))
            
            # Add the current user message
            messages.append(HumanMessage(content=request.query))

            inputs = {
                "messages": messages,
                "original_query": request.query,
                "rag_query": "",
                "search_query": "",
                "is_out_of_scope": False,
                "retrieved_docs": "",
                "search_results": "",
                "filtered_context": "",
                "final_answer": "",
            }
            
            # Send initial step
            yield f"data: {json.dumps({'type': 'step', 'step': 'Initializing...', 'session_id': session_id})}\n\n"
            
            # Import nodes for manual execution to enable token streaming
            from .graph import plan_node, retrieve_and_search_node, critique_node, out_of_scope_node
            
            # Execute planning
            yield f"data: {json.dumps({'type': 'step', 'step': 'Planning query analysis...', 'session_id': session_id})}\n\n"
            plan_result = plan_node(inputs)
            inputs.update(plan_result)
            
            # Check if out of scope
            if inputs["is_out_of_scope"]:
                yield f"data: {json.dumps({'type': 'step', 'step': 'Handling out-of-scope query...', 'session_id': session_id})}\n\n"
                final_result = out_of_scope_node(inputs)
                inputs.update(final_result)
                
                # Stream the out-of-scope message token by token
                yield f"data: {json.dumps({'type': 'streaming_start', 'session_id': session_id})}\n\n"
                for char in inputs['final_answer']:
                    yield f"data: {json.dumps({'type': 'token', 'content': char, 'session_id': session_id})}\n\n"
                    await asyncio.sleep(0.02)  # Small delay for visual effect
                
                yield f"data: {json.dumps({'type': 'streaming_end', 'answer': inputs['final_answer'], 'session_id': session_id})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
                return
            
            # Execute retrieval and search
            yield f"data: {json.dumps({'type': 'step', 'step': 'Retrieving from knowledge base and searching...', 'session_id': session_id})}\n\n"
            retrieve_result = await retrieve_and_search_node(inputs)
            inputs.update(retrieve_result)
            
            # Execute critique and filter
            yield f"data: {json.dumps({'type': 'step', 'step': 'Filtering and analyzing context...', 'session_id': session_id})}\n\n"
            critique_result = critique_node(inputs)
            inputs.update(critique_result)
            
            # Execute streaming writer with token-by-token output
            yield f"data: {json.dumps({'type': 'step', 'step': 'Generating response...', 'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'type': 'streaming_start', 'session_id': session_id})}\n\n"
            
            # Create the writer prompt and chain
            from langchain_core.prompts import ChatPromptTemplate
            from .graph import llm
            
            prompt = ChatPromptTemplate.from_template(
                """You are a final answer synthesizer. Craft a comprehensive, well-structured answer using the provided 'Filtered Context'.
                 If the context is empty, inform the user you couldn't find relevant information.
                 Include chat history context if relevant to provide a conversational flow.

                 Chat History: {chat_history}
                 User's Original Query: {original_query}
                 Filtered Context: {filtered_context}"""
            )
            
            # Format chat history for the prompt
            chat_history_str = ""
            if inputs.get("messages"):
                chat_history_str = "\n".join([
                    f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
                    for msg in inputs["messages"][-5:]  # Include last 5 messages for context
                ])
            
            chain = prompt | llm
            
            # Stream the response token by token
            accumulated_answer = ""
            async for chunk in chain.astream({
                "original_query": inputs["original_query"], 
                "filtered_context": inputs["filtered_context"],
                "chat_history": chat_history_str
            }):
                if chunk.content:
                    accumulated_answer += chunk.content
                    # Send each character for smooth streaming effect
                    for char in chunk.content:
                        yield f"data: {json.dumps({'type': 'token', 'content': char, 'session_id': session_id})}\n\n"
                        await asyncio.sleep(0.01)  # Small delay for visual effect
            
            # Send the complete answer
            yield f"data: {json.dumps({'type': 'streaming_end', 'answer': accumulated_answer, 'session_id': session_id})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
            
        except Exception as e:
            print(f"Error in token stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'session_id': session_id})}\n\n"

    return StreamingResponse(
        generate_token_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@api.post("/chat-stream-native")
async def chat_stream_native_endpoint(request: ChatRequest):
    """
    Native LangGraph streaming endpoint using add reduction for messages.
    This endpoint demonstrates the proper use of the add operator.
    """
    session_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": session_id}}

    async def generate_native_stream():
        try:
            # Convert chat history from dicts to BaseMessage objects
            messages = []
            for msg in request.chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") in ("assistant", "ai"):
                    messages.append(AIMessage(content=msg.get("content", "")))
            
            # Add the current user message
            current_message = HumanMessage(content=request.query)
            messages.append(current_message)

            # With add reduction, we can send just the new message and it will be added to existing state
            inputs = {
                "messages": [current_message],  # Only the new message, thanks to add reduction!
                "original_query": request.query,
                "rag_query": "",
                "search_query": "",
                "is_out_of_scope": False,
                "retrieved_docs": "",
                "search_results": "",
                "filtered_context": "",
                "final_answer": "",
            }
            
            # If there's chat history, we need to initialize the state first
            if len(messages) > 1:
                # Initialize with all messages for the first time
                init_inputs = {
                    "messages": messages,
                    "original_query": request.query,
                    "rag_query": "",
                    "search_query": "",
                    "is_out_of_scope": False,
                    "retrieved_docs": "",
                    "search_results": "",
                    "filtered_context": "",
                    "final_answer": "",
                }
                inputs = init_inputs
            
            # Send initial step
            yield f"data: {json.dumps({'type': 'step', 'step': 'Initializing native LangGraph streaming...', 'session_id': session_id})}\n\n"
            
            current_step = "planning"
            
            # Stream the graph execution using LangGraph's native streaming
            async for chunk in graph_app.astream(inputs, config=thread_config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if node_name == "planner":
                        if node_output.get("is_out_of_scope"):
                            yield f"data: {json.dumps({'type': 'step', 'step': 'Query is out of scope...', 'session_id': session_id})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'step', 'step': 'Planning completed, retrieving information...', 'session_id': session_id})}\n\n"
                    
                    elif node_name == "retrieve_and_search":
                        yield f"data: {json.dumps({'type': 'step', 'step': 'Information retrieved, analyzing context...', 'session_id': session_id})}\n\n"
                    
                    elif node_name == "critique":
                        yield f"data: {json.dumps({'type': 'step', 'step': 'Context analyzed, generating response...', 'session_id': session_id})}\n\n"
                    
                    elif node_name in ["writer", "out_of_scope"]:
                        if "final_answer" in node_output:
                            final_answer = node_output["final_answer"]
                            yield f"data: {json.dumps({'type': 'answer', 'answer': final_answer, 'session_id': session_id})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
            
        except Exception as e:
            print(f"Error in native LangGraph stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'session_id': session_id})}\n\n"

    return StreamingResponse(
        generate_native_stream(),
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
    session_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": session_id}}
    
    # Convert chat history from dicts to BaseMessage objects
    messages = []
    for msg in request.chat_history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") in ("assistant", "ai"):
            messages.append(AIMessage(content=msg.get("content", "")))
    
    # Add the current user message
    messages.append(HumanMessage(content=request.query))

    inputs = {
        "messages": messages,
        "original_query": request.query,
        "rag_query": "",
        "search_query": "",
        "is_out_of_scope": False,
        "retrieved_docs": "",
        "search_results": "",
        "filtered_context": "",
        "final_answer": "",
    }
    
    # Asynchronously invoke the LangGraph agent
    final_state = await graph_app.ainvoke(inputs, config=thread_config)
    
    return {"answer": final_state.get("final_answer", "Sorry, something went wrong.")}

@api.get("/")
def read_root():
    return {"message": "Welcome to the Space-GPT API. Go to /docs for the API documentation."}