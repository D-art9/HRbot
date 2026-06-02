import os

from dotenv import load_dotenv
from langchain_chroma import Chroma

from core.embedding_provider import get_embedding_function

load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(PROJECT_ROOT, "data", "chroma_db"))

_vector_store = None

def get_vector_store(collection_name="hr_knowledge_base"):
    global _vector_store
    if _vector_store is None:
        print("INITIALIZING VECTOR STORE (ONCE)")
        embeddings = get_embedding_function()

        # Ensure the directory exists
        if not os.path.exists(CHROMA_PATH):
            os.makedirs(CHROMA_PATH)

        _vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    return _vector_store

def clear_vector_store_cache():
    global _vector_store
    _vector_store = None
