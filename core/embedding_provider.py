from langchain_huggingface import HuggingFaceEmbeddings

_embedding_function = None

def get_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        print("INITIALIZING EMBEDDING MODEL (ONCE)")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        _embedding_function = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    return _embedding_function
