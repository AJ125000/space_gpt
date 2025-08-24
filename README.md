# Space-GPT: AI-Powered Space Research Assistant

Space-GPT is a sophisticated AI-powered research assistant specializing in space, astronomy, and astrophysics. It leverages a powerful graph-based architecture to understand user queries, retrieve information from a specialized knowledge base, search the web for real-time data, and synthesize comprehensive, accurate answers.

## ✨ Features

- **Advanced RAG Pipeline**: Combines knowledge base retrieval with real-time web search for up-to-date and contextually relevant answers
- **Intelligent Query Planning**: Analyzes user intent to determine whether to query the local knowledge base, search the web, or handle out-of-scope questions
- **Self-Critiquing Mechanism**: Filters and refines retrieved information to ensure only the most relevant context is used for generating answers
- **Interactive Web Frontend**: A sleek, ChatGPT-style user interface with real-time step updates
- **Real-time Processing Steps**: The frontend displays the current processing step of the backend graph, providing transparency to the user
- **Chat History Support**: Maintains conversation context across multiple exchanges
- **Full Observability**: Integrated with **LangSmith** for end-to-end tracing and debugging of the entire graph execution
- **Modular Architecture**: Built with FastAPI, LangGraph, and LlamaIndex, making it easy to extend and maintain

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python
- **Orchestration**: LangGraph, Langchain
- **LLM**: Google Gemini 2.5 Flash
- **Vector Database**: Qdrant Cloud
- **Search**: Serper API for Google Search
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Observability**: LangSmith
- **Document Processing**: LlamaIndex with Gemini embeddings

## 🏗️ Architecture

The application is built around a LangGraph-powered agent that follows these steps:

1. **Planner**: Analyzes the user's query to create a retrieval plan. It decides if the query is space-related and formulates specific search queries for the knowledge base and the web.
2. **Retrieve & Search**: Concurrently fetches documents from the Qdrant vector store and performs a web search using the Serper API.
3. **Critique**: Examines all retrieved information, filters out irrelevant content, and synthesizes a clean, focused context.
4. **Writer**: Uses the filtered context to generate a final, comprehensive answer for the user.

## 📂 Project Structure

```
.
├── app/                    # FastAPI application and core logic
│   ├── __init__.py
│   ├── config.py           # Environment settings and configuration
│   ├── graph.py            # LangGraph definition and node implementations
│   ├── main.py             # FastAPI endpoints (REST and streaming)
│   └── schemas.py          # Pydantic models for API and graph state     
├── core/                   # Core logic for retrieval and tools
│   ├── __init__.py
│   ├── retriever.py        # Knowledge base retrieval with LlamaIndex
│   └── tools.py            # Web search tool using Serper API
├── ingestion/              # Data ingestion pipeline
│   ├── __init__.py
│   ├── pipeline.py         # Script to process and store documents
│   └── documents/          # Source text files for the knowledge base
├── storage/                # Local storage for LlamaIndex artifacts
│   └── summary_index/      # Vectorized document index
├── .env.sample             # Template for environment variables
├── index.html              # Main frontend interface
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An account with [Google AI Studio](https://aistudio.google.com/) for a Gemini API key
- An account with [Qdrant Cloud](https://cloud.qdrant.io/) for a vector database
- An account with [Serper](https://serper.dev/) for a web search API key
- An account with [LangSmith](https://smith.langchain.com/) for tracing (optional but recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/AJ125000/space_gpt.git
cd space_gpt
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root by copying the sample file:

```bash
cp .env.sample .env
```

Now, open the `.env` file and add your API keys and credentials:

```env
# Google Gemini API Key
GOOGLE_API_KEY="your_google_api_key"

# Serper API Key for Web Search
SERPER_API_KEY="your_serper_api_key"

# Qdrant Cloud Configuration
QDRANT_URL="your_qdrant_url"
QDRANT_API_KEY="your_qdrant_api_key"

# LangSmith Configuration (Optional)
LANGSMITH_TRACING_V2="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your_langsmith_api_key"
LANGSMITH_PROJECT="Space-GPT"
```

### 5. Ingest Data into the Knowledge Base

Before running the application, you need to populate the Qdrant vector database with your specialized documents.

Run the ingestion pipeline:
```bash
python -m ingestion.pipeline
```
This script will read the text files from `ingestion/documents/`, process them, and store the embeddings in your Qdrant collection.

## 🏃‍♀️ Running the Application

### 1. Start the Backend Server

```bash
uvicorn app.main:api --reload
```
The API will be available at `http://localhost:8000`. You can access the documentation at `http://localhost:8000/docs`.

### 2. Launch the Frontend

Open the `index.html` file in your web browser. It will connect to the backend API automatically.

You can now start chatting with your Space-GPT assistant!

## 🧪 Testing

The repository includes several testing utilities:


## 📡 API Endpoints

- **POST `/chat`**: Standard chat endpoint (returns final answer only)
- **POST `/chat-stream`**: Streaming chat endpoint (returns real-time step updates)
- **GET `/`**: Health check endpoint
- **GET `/docs`**: Interactive API documentation

### Chat Request Format

```json
{
  "query": "What is the James Webb Space Telescope?",
  "chat_history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ]
}
```

## 🔧 Technologies Used

- **FastAPI**: For building the REST API and Server-Sent Events endpoints
- **LangChain & LangGraph**: For building the AI agent and graph-based orchestration
- **LlamaIndex**: For document indexing, retrieval, and vector storage management
- **Qdrant**: For vector storage and similarity search
- **Google Gemini**: Large language model for reasoning and text generation
- **Serper API**: For real-time web search capabilities
- **LangSmith**: For observability and debugging of LLM applications
- **Trafilatura**: For downloading, parsing, and extracting text, metadata, and comments from web pages

## 🎯 Key Features Explained

### Real-time Step Updates
The application provides real-time feedback to users about what's happening during processing:
- Planning query analysis
- Retrieving from knowledge base
- Searching external sources
- Filtering and analyzing context
- Generating final response

### Intelligent Query Routing
The system automatically determines the best approach for each query:
- **In-scope**: Space, astronomy, astrophysics queries → Full RAG pipeline
- **Out-of-scope**: Non-space topics → Polite decline with explanation

### Hybrid Information Retrieval
Combines two information sources for comprehensive answers:
- **Local Knowledge Base**: Curated space documents with vector similarity search
- **Web Search**: Real-time information from Google Search via Serper API

## 🚀 Adding New Documents

To expand the knowledge base:

1. Add new text files to `ingestion/documents/`
2. Run the ingestion pipeline: `python -m ingestion.pipeline`
3. The new documents will be processed and added to the vector database

## 🐛 Troubleshooting

### Common Issues

**API Key Errors**: Make sure all required API keys are set in your `.env` file
**Vector Database Connection**: Verify your Qdrant URL and API key are correct
**Missing Dependencies**: Run `pip install -r requirements.txt` to ensure all packages are installed
**LangSmith Not Working**: Check that `LANGSMITH_API_KEY` is set correctly in your `.env` file

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.
