import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings(BaseSettings):
    # This configuration tells Pydantic to load from the .env file for local dev,
    # but actual environment variables will always take precedence in production.
    model_config = SettingsConfigDict(env_file=os.path.join(ROOT_DIR, ".env"), extra="ignore")

    # --- API Keys & Secrets (Loaded from environment) ---
    google_api_key: str
    langsmith_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    langsmith_tracing_v2: str
    serper_api_key: str
    langsmith_endpoint: str
    langsmith_project: str
    
    # --- Local Storage Paths (for non-vector data) ---
    # The summary index is stored locally as it's not part of the vector DB.
    SUMMARY_INDEX_DIR: str = os.path.join(ROOT_DIR, "storage", "summary_index")

settings = Settings()