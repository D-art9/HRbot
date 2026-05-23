import os
import sys
import asyncio
import time

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flashrank import Ranker, RerankRequest
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm_provider import llm
from core.vector_store import get_vector_store, clear_vector_store_cache
from core.logger import logger

DOCUMENT_PATH = "data/documents"

# Lazy-loaded ranker (initialized on first use, not at import time)
_ranker = None

def get_ranker():
    """Lazy-load the ranker on first use to avoid startup delay."""
    global _ranker
    if _ranker is None:
        logger.info("RANKER: Initializing ranker model (first use)")
        print("[RANKER] Initializing ms-marco-MiniLM-L-12-v2 model...")
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="data/models")
        logger.info("RANKER: Model initialization complete")
        print("[RANKER] Model ready")
    return _ranker

# Global chat history for conversational memory
chat_history = []


def load_documents():
    """
    Loads documents from the data/documents directory.
    Supports PDF, DOCX, and TXT. Cleans metadata to keep only essential fields.
    """
    print("\n--- STAGE 1: DOCUMENT INGESTION ---")
    all_documents = []

    if not os.path.exists(DOCUMENT_PATH):
        print(f"Error: Document path {DOCUMENT_PATH} does not exist.")
        return []

    file_count = 0
    for root, dirs, files in os.walk(DOCUMENT_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                elif file.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                elif file.endswith(".txt"):
                    loader = TextLoader(file_path)
                    docs = loader.load()
                else:
                    continue

                # Clean metadata and retain only essential fields
                for doc in docs:
                    old_metadata = doc.metadata
                    clean_metadata = {
                        "source_file": file,
                        "relative_path": os.path.relpath(file_path, DOCUMENT_PATH),
                        "category": os.path.relpath(file_path, DOCUMENT_PATH).split(
                            os.sep
                        )[0],
                        "page": old_metadata.get("page", 0) + 1,
                    }
                    doc.metadata = clean_metadata

                all_documents.extend(docs)
                file_count += 1
                print(f"  [+] Loaded: {file}")

            except Exception as e:
                print(f"  [!] Error loading {file}: {e}")

    print(f"Total files ingested: {file_count}")
    return all_documents


def split_documents(documents):
    """
    Splits documents into smaller chunks for vector storage.
    """
    print("\n--- STAGE 2: CHUNKING ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Sequence tracking for windowed retrieval
    file_chunk_counters = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file")
        if source not in file_chunk_counters:
            file_chunk_counters[source] = 0
        chunk.metadata["chunk_index"] = file_chunk_counters[source]
        file_chunk_counters[source] += 1

    print(f"Generated {len(chunks)} chunks with sequence metadata.")
    return chunks


def initialize_rag():
    """
    Main initialization function: Loads, splits, and stores documents.
    """
    documents = load_documents()
    if not documents:
        print("No documents found to index.")
        return

    chunks = split_documents(documents)

    print("\n--- STAGE 3: VECTOR STORAGE ---")
    vectordb = get_vector_store(collection_name="hr_knowledge_base")

    print("Clearing existing vector store...")
    try:
        vectordb.delete_collection()
        clear_vector_store_cache()
        vectordb = get_vector_store(collection_name="hr_knowledge_base")
    except Exception as e:
        print(f"Note: Could not clear collection: {e}")

    print("Indexing documents into ChromaDB...")
    vectordb.add_documents(chunks)
    print(f"Successfully indexed {len(chunks)} chunks.")


def rewrite_query(query):
    """
    Rewrites a conversational follow-up question into a standalone query.
    """
    if not chat_history:
        return query

    if llm is None:
        return query

    rephrase_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
        Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the conversation history.
        If the follow-up question is already a standalone question, return it as is.

        Chat History:
        {chat_history}

        Follow-up Question: {question}
        Standalone Question:
        """,
    )

    history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history[-3:]])
    final_prompt = rephrase_prompt.format(chat_history=history_str, question=query)

    try:
        response = llm.invoke(final_prompt)
        return response.content.strip()
    except Exception as e:
        print(f"  [!] Error rewriting query: {e}")
        return query


def retrieve_with_scores(query, k=5):
    """
    Retrieves documents with their similarity scores.
    """
    vectordb = get_vector_store(collection_name="hr_knowledge_base")
    results = vectordb.similarity_search_with_score(query, k=k)
    return results


def query_rag(query, k=5):
    """
    Searches the vector store with MMR and expansions.
    """
    vectordb = get_vector_store(collection_name="hr_knowledge_base")

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.6,
        },
    )
    initial_results = retriever.invoke(query)

    expanded_results = []
    seen_contents = set()

    for doc in initial_results:
        if doc.page_content not in seen_contents:
            expanded_results.append(doc)
            seen_contents.add(doc.page_content)

        source = doc.metadata.get("source_file")
        current_idx = doc.metadata.get("chunk_index")

        if source and current_idx is not None:
            neighbor_data = vectordb.get(
                where={
                    "$and": [
                        {"source_file": {"$eq": source}},
                        {"chunk_index": {"$eq": current_idx + 1}},
                    ]
                }
            )

            if neighbor_data and neighbor_data.get("documents"):
                neighbor_doc = Document(
                    page_content=neighbor_data["documents"][0],
                    metadata=neighbor_data["metadatas"][0],
                )
                if neighbor_doc.page_content not in seen_contents:
                    expanded_results.append(neighbor_doc)
                    seen_contents.add(neighbor_doc.page_content)

    return expanded_results


def rerank_results(query, documents):
    """
    Uses a Cross-Encoder to re-rank the retrieved chunks based on
    how well they actually answer the specific question.
    """
    print("\n[RAG] Starting re-ranking...")
    
    if not documents:
        logger.warning("RAG: No documents to rerank")
        print("[RAG] ⚠ No documents provided for reranking")
        return []

    # Prepare data for FlashRank
    passages = []
    for i, doc in enumerate(documents):
        passages.append({"id": i, "text": doc.page_content, "meta": doc.metadata})

    logger.info(f"RAG: Re-ranking {len(passages)} passages")
    print(f"[RAG] Re-ranking {len(passages)} passages...")
    
    # Execute Re-ranking with lazy-loaded ranker
    try:
        ranker = get_ranker()
        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)
        logger.info(f"RAG: Re-ranking complete")
        print(f"[RAG] ✓ Re-ranking complete")
    except Exception as e:
        logger.error(f"RAG: Reranking failed: {e}")
        print(f"[RAG] ✗ Reranking error: {e}")
        # Return original docs if reranking fails
        return documents

    # Convert back to LangChain Documents, now sorted by "Relevancy Score"
    reranked_docs = []
    for res in results:
        # res['score'] is the new Cross-Encoder score (0.0 to 1.0)
        new_doc = Document(page_content=res["text"], metadata=res["meta"])
        new_doc.metadata["rerank_score"] = res["score"]
        reranked_docs.append(new_doc)

    return reranked_docs


def ask_question(query):
    """
    Generates a professional HR response using query rewriting,
    retrieval, and cross-encoder re-ranking for maximum accuracy.
    
    CRITICAL: This function includes timeout protection to prevent hanging.
    """
    logger.info(f"RAG: Processing query: {query[:100]}...")
    print("\n--- STAGE 5: PROFESSIONAL HR RESPONSE GENERATION ---")
    print(f"[RAG] Query: {query}")

    # Step 1 — Conversational Query Rewriting
    print("[RAG] Step 1: Rewriting query...")
    rewritten_query = rewrite_query(query)
    logger.info(f"RAG: Rewritten query: {rewritten_query[:100]}")
    print(f"[RAG] ✓ Rewritten Query: {rewritten_query}")

    # Step 2 — Initial Retrieval from Chroma (Increased k for re-ranking)
    # We pull more docs (15) so the re-ranker has a good pool to choose from
    print("[RAG] Step 2: Retrieving documents from ChromaDB...")
    initial_docs = query_rag(rewritten_query, k=15)
    logger.info(f"RAG: Retrieved {len(initial_docs)} initial documents")
    print(f"[RAG] ✓ Retrieved {len(initial_docs)} documents")

    if not initial_docs:
        logger.warning("RAG: No documents found in vector store")
        print("[RAG] ✗ No documents found")
        return "I could not find any relevant information in the company documents.", []

    # Step 3 — Smart Re-ranking (The Cross-Encoder Step)
    print("[RAG] Step 3: Re-ranking retrieved documents...")
    reranked_docs = rerank_results(rewritten_query, initial_docs)
    logger.info(f"RAG: Re-ranked {len(reranked_docs)} documents")
    print(f"[RAG] ✓ Re-ranked {len(reranked_docs)} documents")

    # Step 4 — Enhanced Hallucination Protection
    # Cross-Encoder scores are 0 to 1 (1 is perfect).
    # If the BEST chunk has a score < 0.35, we don't trust the context.
    best_rerank_score = (
        reranked_docs[0].metadata["rerank_score"] if reranked_docs else 0
    )
    logger.info(f"RAG: Best rerank score: {best_rerank_score:.4f}")
    print(f"[RAG] Rerank Relevancy Score: {best_rerank_score:.4f}")

    if best_rerank_score < 0.35:
        logger.warning(f"RAG: Score {best_rerank_score} below confidence threshold")
        print(f"[RAG] ⚠ Score below confidence threshold")
        return (
            "I could not find sufficiently reliable information in the company documents to answer this question confidently.",
            [],
        )

    # Step 5 — Context Assembly
    # We only send the top 5 BEST re-ranked chunks to the LLM
    final_context_docs = reranked_docs[:5]
    context = "\n\n".join([doc.page_content for doc in final_context_docs])
    logger.info(f"RAG: Assembled context from {len(final_context_docs)} documents")
    print(f"[RAG] ✓ Context assembled ({len(context)} chars)")

    # Step 6 — Professional prompt with history awareness
    history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history[-3:]])

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""
You are a professional enterprise HR assistant AI for a multinational company.

Your responsibilities:
- Answer professionally and clearly
- Maintain a formal corporate tone
- Only answer using the provided context and history
- If information is unavailable, clearly state it
- Do not speculate or hallucinate

Conversation History:
{history}

Context:
{context}

Employee Question:
{question}

Professional HR Response:
    Respond using the following enterprise HR structure:

    1. Policy Summary
    - Brief explanation of the policy

    2. Eligibility
    - Who the policy applies to

    3. Important Conditions
    - Key rules, limitations, timelines, or restrictions

    4. Compliance Notes
    - Any warnings, obligations, or governance requirements

    5. Sources
    - Mention the referenced company documents naturally

    Professional HR Response:
""",
    )

    final_prompt = prompt.format(history=history_str, context=context, question=query)

    if llm is None:
        logger.error("RAG: LLM is None")
        print("[RAG] ✗ LLM not initialized")
        return "Error: LLM not initialized.", final_context_docs

    try:
        print("[RAG] Step 6: Calling LLM (with 30s timeout)...")
        logger.info("RAG: Invoking LLM for response generation")
        
        # Add timeout protection to prevent hanging
        start_time = time.time()
        response = llm.invoke(final_prompt)
        elapsed_time = time.time() - start_time
        
        logger.info(f"RAG: LLM responded in {elapsed_time:.2f}s")
        print(f"[RAG] ✓ LLM response received ({elapsed_time:.2f}s)")
        
        answer = response.content
        chat_history.append((query, answer))
        
        logger.info("RAG: Response complete, ready to stream")
        print(f"[RAG] ✓ Response ready for streaming ({len(answer)} chars)")
        print("[RAG] STREAM READY TO YIELD")
        
        return answer, final_context_docs
        
    except asyncio.TimeoutError:
        error_msg = "LLM call timed out (exceeded 30 seconds)"
        logger.error(f"RAG: {error_msg}")
        print(f"[RAG] ✗ {error_msg}")
        return f"Error: {error_msg}", final_context_docs
        
    except Exception as e:
        logger.error(f"RAG: LLM error: {str(e)[:200]}")
        print(f"[RAG] ✗ LLM error: {e}")
        return f"Error: {e}", final_context_docs


if __name__ == "__main__":
    initialize_rag()

    test_queries = [
        "What is the leave policy?",
        "How can I apply for it?",
        "What about remote work?",
    ]

    for q in test_queries:
        ans, docs = ask_question(q)
        print("\n" + "=" * 60)
        print(f"QUESTION: {q}".center(60))
        print("=" * 60)
        print(f"\nHR RESPONSE:\n{ans}")
        print("\nSOURCES:")
        for i, d in enumerate(docs, start=1):
            print(
                f"  {i}. {d.metadata.get('source_file')} (Page: {d.metadata.get('page')})"
            )
        print("-" * 60)
