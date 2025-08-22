# Space-Bot

Space-Bot is an AI-powered space research assistant that can answer questions about space, astronomy, and astrophysics. It uses a combination of a local knowledge base and web search to provide comprehensive and accurate answers.

## Features

- **Conversational AI:** A friendly chat interface for interacting with the assistant.
- **Knowledge Base:** A collection of documents on various space-related topics.
- **Web Search:** The ability to search the web for the latest information.
- **Advanced Query Processing:** Uses a graph-based approach to understand queries, retrieve relevant information, and generate high-quality answers.

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration settings, including API keys and paths
│   ├── graph.py            # The core LangGraph application logic
│   ├── main.py             # FastAPI application entry point
│   └── schemas.py          # Pydantic schemas for API requests and graph state
├── core/
│   ├── __init__.py
│   ├── retriever.py        # Knowledge base retrieval logic
│   └── tools.py            # Web search tool integration
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py         # Script for ingesting documents into the knowledge base
│   └── documents/          # Directory for storing knowledge base documents
├── storage/
│   └── summary_index/      # Local storage for the summary index
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AJ125000/space_gpt.git
    cd space-bot
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add the following:
    ```
    GOOGLE_API_KEY="your_google_api_key"
    LANGCHAIN_API_KEY="your_langchain_api_key"
    QDRANT_URL="your_qdrant_url"
    QDRANT_API_KEY="your_qdrant_api_key"
    ```

## Usage

1.  **Run the FastAPI application:**
    ```bash
    uvicorn app.main:api --reload
    ```

2.  **Access the API documentation:**
    Open your browser and go to `http://127.0.0.1:8000/docs`.

3.  **Send a chat request:**
    Use the `/chat` endpoint to send a query to the assistant. The request body should be a JSON object with the following structure:
    ```json
    {
      "query": "What is a black hole?",
      "chat_history": []
    }
    ```

## Ingestion

To add new documents to the knowledge base, place them in the `ingestion/documents` directory and run the ingestion pipeline:

```bash
python -m ingestion.pipeline
```

## Technologies Used

- **FastAPI:** For building the web API.
- **LangChain & LangGraph:** For building the AI agent and graph-based logic.
- **LlamaIndex:** For data indexing and retrieval.
- **Qdrant:** For vector storage and similarity search.
- **Gemini:** Google's family of generative AI models.
- **DuckDuckGo Search:** For web search.
