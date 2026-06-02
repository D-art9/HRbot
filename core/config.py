# Central configuration management
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Cyvia observability (optional — tracing disabled if CYVIA_API_KEY unset)
    CYVIA_API_KEY = os.getenv("CYVIA_API_KEY")
    CYVIA_BASE_URL = os.getenv("CYVIA_BASE_URL", "https://app.cyvia.ai/api")

    # Model Configuration
    # Defaulting to a Groq model if not specified, as the provider uses ChatGroq
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

    # Paths
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
    DOCS_PATH = os.getenv("DOCS_PATH", "data/documents")

    # Embeddings
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )


settings = Settings()
