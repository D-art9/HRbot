from langchain_community.embeddings import FastEmbedEmbeddings

_embedding_function = None

def get_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        print("INITIALIZING EMBEDDING MODEL (ONCE) WITH FASTEMBED (ONNX)")
        _embedding_function = FastEmbedEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir="data/models"
        )
    return _embedding_function
