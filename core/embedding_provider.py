from langchain_community.embeddings import FastEmbedEmbeddings

_embedding_function = None

def get_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        print("INITIALIZING EMBEDDING MODEL (ONCE) WITH FASTEMBED (ONNX)")
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cache_dir = os.path.join(project_root, "data", "models")
        _embedding_function = FastEmbedEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=cache_dir
        )
    return _embedding_function
