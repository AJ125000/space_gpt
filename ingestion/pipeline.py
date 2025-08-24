import os
import glob
from llama_index.core import Document, Settings, VectorStoreIndex, SummaryIndex, StorageContext
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client.http import models as qmodels  # for explicit create

# Import settings from the centralized config file
from app.config import settings

# Change this if your folder is named differently
ARTICLES_DIR = os.path.join(os.path.dirname(__file__), "documents")

def _embedding_dim():
    # probe once to get dimension; avoids guessing
    vec = Settings.embed_model.get_text_embedding("dimension probe")
    return len(vec)

def main():
    os.makedirs(settings.SUMMARY_INDEX_DIR, exist_ok=True)

    # Use Google GenAI free embedding model
    print("Using Google's free 'embedding-001' model.")
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/embedding-001",
        api_key=settings.google_api_key,
    )

    # Load .txt files
    docs = []
    for path in glob.glob(os.path.join(ARTICLES_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        docs.append(Document(text=text, metadata={"source": os.path.basename(path)}))

    if not docs:
        print(f"[Ingest] No .txt files found in {ARTICLES_DIR}. Please add documents and re-run.")
        return

    print(f"[Ingest] Loaded {len(docs)} documents.")

    # Qdrant client
    print("[Ingest] Setting up Qdrant Cloud client...")
    client = qdrant_client.QdrantClient(
        url=settings.qdrant_url,  # e.g., https://<cluster-id>.cloud.qdrant.io:6333
        api_key=settings.qdrant_api_key,
    )

    collection_name = "space_gpt"

    # Ensure collection exists with correct vector size & metric
    dim = _embedding_dim()
    try:
        client.get_collection(collection_name)
        print(f"[Ingest] Found existing collection '{collection_name}'.")
    except Exception:
        print(f"[Ingest] Creating collection '{collection_name}' (size={dim}, COSINE).")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    # Attach Qdrant via StorageContext (CRITICAL change)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build vector index in Qdrant
    VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)
    info = client.get_collection(collection_name)
    print(f"[Ingest] Qdrant collection '{collection_name}' vectors: {info.points_count}")

    # Persist local summary index
    s_index = SummaryIndex.from_documents(docs)
    s_index.storage_context.persist(persist_dir=settings.SUMMARY_INDEX_DIR)
    print(f"[Ingest] SummaryIndex persisted at: {settings.SUMMARY_INDEX_DIR}")

if __name__ == "__main__":
    main()