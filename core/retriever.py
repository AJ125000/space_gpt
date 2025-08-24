import os
from typing import List, Dict, Any, Tuple
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage, SummaryIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

from app.config import settings

class KnowledgeBase:
    def __init__(self):
        print("Initializing KnowledgeBase...")
        
        # Configure LlamaIndex to use Google GenAI for both LLM and embeddings
        print("Configuring Google GenAI models for LlamaIndex...")
        Settings.llm = GoogleGenAI(
            model ="models/gemini-1.5-flash-latest", api_key=settings.google_api_key
        )
        Settings.embed_model = GoogleGenAIEmbedding(
            model_name="models/embedding-001", api_key=settings.google_api_key
        )

        # Connect to Qdrant Cloud
        print("Connecting to Qdrant Cloud...")
        client = qdrant_client.QdrantClient(
            url=settings.qdrant_url, 
            api_key=settings.qdrant_api_key
        )
        collection_name = "space_gpt"
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        
        self._vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        print("Successfully connected to Qdrant and loaded vector index.")

        # Load local SummaryIndex
        summary_dir = settings.SUMMARY_INDEX_DIR
        if os.path.isdir(summary_dir) and os.listdir(summary_dir):
            print(f"Loading summary index from: {summary_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=summary_dir)
            self._summary_index = load_index_from_storage(storage_context)
        else:
            print("Warning: No local summary index found. Summarization may be limited.")
            self._summary_index = SummaryIndex.from_documents([])

        self._kb_retriever = self._vector_index.as_retriever(similarity_top_k=2)
        self._summary_engine = self._summary_index.as_query_engine(
            response_mode="tree_summarize", use_async=True
        )

    async def retrieve(self, query: str) -> str:
        # Retrieve nodes from the vector store
        nodes = self._kb_retriever.retrieve(query)
        
        # Format the retrieved chunks
        chunks = []
        for i, n in enumerate(nodes, start=1):
            chunks.append(f"[KB:{i}] {n.get_text()}")
        
        kb_context_text = "\n\n".join(chunks) if chunks else "(No relevant information found in the knowledge base)"

        # Asynchronously get the summary abstract
        try:
            summary_resp = await self._summary_engine.aquery(query)
            if summary_resp and str(summary_resp).strip():
                kb_context_text += "\n\n" + f"[KB:abstract] {summary_resp}"
        except Exception as e:
            print(f"Error during summary query: {e}")
            pass

        return kb_context_text

# Create a single instance to be used by the application
knowledge_base = KnowledgeBase()