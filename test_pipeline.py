#!/usr/bin/env python3
"""
Comprehensive pipeline test for debugging the streaming issue.

This script tests:
1. Model initialization (embeddings, ranker, LLM)
2. RAG pipeline (retrieval + reranking)
3. Response generation
4. Streaming format

Run with: python test_pipeline.py
"""

import os
import sys
import asyncio
import json
import time

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.logger import logger
from core.embedding_provider import get_embedding_function
from core.vector_store import get_vector_store
from core.llm_provider import llm
from modules.rag_module import (
    ask_question, 
    get_ranker, 
    rerank_results,
    query_rag
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}".center(70))
    print("="*70)


def test_model_initialization():
    """Test 1: Verify all models initialize correctly."""
    print_section("TEST 1: Model Initialization")
    
    try:
        print("✓ Testing embedding model...")
        embeddings = get_embedding_function()
        assert embeddings is not None
        print("  ✓ Embeddings loaded")
        
        print("\n✓ Testing vector store...")
        vectordb = get_vector_store(collection_name="hr_knowledge_base")
        assert vectordb is not None
        print("  ✓ Vector store loaded")
        
        print("\n✓ Testing ranker model...")
        ranker = get_ranker()
        assert ranker is not None
        print("  ✓ Ranker loaded")
        
        print("\n✓ Testing LLM provider...")
        assert llm is not None, "LLM not initialized"
        print("  ✓ LLM provider ready")
        
        print("\n✅ All models initialized successfully!")
        logger.info("TEST: Model initialization successful")
        return True
        
    except Exception as e:
        print(f"\n❌ Model initialization failed: {e}")
        logger.error(f"TEST: Model initialization error: {e}")
        return False


def test_rag_pipeline():
    """Test 2: Verify RAG pipeline works end-to-end."""
    print_section("TEST 2: RAG Pipeline")
    
    try:
        test_query = "What is the leave policy?"
        print(f"Query: {test_query}\n")
        
        print("Step 1: Retrieving documents...")
        start_time = time.time()
        docs = query_rag(test_query, k=5)
        elapsed = time.time() - start_time
        print(f"  ✓ Retrieved {len(docs)} documents ({elapsed:.2f}s)")
        
        if not docs:
            print("  ⚠ No documents found in vector store")
            print("    Hint: Run data/documents/setup.py to index documents")
            return False
        
        print("\nStep 2: Reranking documents...")
        start_time = time.time()
        reranked = rerank_results(test_query, docs)
        elapsed = time.time() - start_time
        print(f"  ✓ Reranked {len(reranked)} documents ({elapsed:.2f}s)")
        
        if reranked:
            best_score = reranked[0].metadata.get("rerank_score", 0)
            print(f"  ✓ Best rerank score: {best_score:.4f}")
        
        print("\nStep 3: Generating response...")
        start_time = time.time()
        answer, sources = ask_question(test_query)
        elapsed = time.time() - start_time
        print(f"  ✓ Response generated ({elapsed:.2f}s)")
        print(f"  ✓ Response length: {len(answer)} characters")
        
        if sources:
            print(f"  ✓ Sources found: {len(sources)}")
            for i, doc in enumerate(sources[:3], 1):
                filename = doc.metadata.get("source_file", "unknown")
                print(f"    {i}. {filename}")
        
        print("\n✅ RAG pipeline working correctly!")
        logger.info("TEST: RAG pipeline successful")
        return True
        
    except Exception as e:
        print(f"\n❌ RAG pipeline failed: {e}")
        logger.error(f"TEST: RAG pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_generation():
    """Test 3: Verify response generation works."""
    print_section("TEST 3: Response Generation")
    
    try:
        test_query = "What are the benefits?"
        print(f"Query: {test_query}\n")
        
        print("Generating response...")
        start_time = time.time()
        answer, sources = ask_question(test_query)
        elapsed = time.time() - start_time
        
        print(f"✓ Response generated in {elapsed:.2f}s")
        print(f"✓ Response length: {len(answer)} characters")
        
        # Check response quality
        if len(answer) < 50:
            print("⚠ Warning: Response seems very short")
        
        if "error" in answer.lower():
            print("⚠ Warning: Response contains error text")
        
        # Preview response
        preview = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"\nResponse preview:\n{preview}")
        
        print("\n✅ Response generation working!")
        logger.info("TEST: Response generation successful")
        return True
        
    except Exception as e:
        print(f"\n❌ Response generation failed: {e}")
        logger.error(f"TEST: Response generation error: {e}")
        return False


async def test_streaming():
    """Test 4: Verify streaming format."""
    print_section("TEST 4: Streaming Format")
    
    try:
        from agents.hr_agent import stream_hr_query
        
        test_query = "What is the policy?"
        print(f"Query: {test_query}\n")
        
        print("Streaming response...\n")
        chunk_count = 0
        token_count = 0
        error_occurred = False
        
        start_time = time.time()
        
        async for chunk in stream_hr_query(test_query, user_id="test_user"):
            chunk_count += 1
            
            # Parse SSE format
            if chunk.startswith("data: "):
                try:
                    json_str = chunk[6:]  # Remove "data: " prefix
                    data = json.loads(json_str.strip())
                    
                    if data.get("type") == "token":
                        token_count += 1
                    elif data.get("type") == "metadata":
                        print(f"  📊 Metadata: {data.get('workflow_stage', 'unknown')}")
                    elif data.get("type") == "done":
                        print(f"  ✅ Stream complete")
                    elif data.get("type") == "error":
                        print(f"  ❌ Error: {data.get('message')}")
                        error_occurred = True
                    
                except json.JSONDecodeError:
                    print(f"  ⚠ Failed to parse: {chunk[:50]}")
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Streamed {chunk_count} chunks")
        print(f"✓ Streamed {token_count} tokens")
        print(f"✓ Total time: {elapsed:.2f}s")
        
        if error_occurred:
            print("\n⚠ Stream had errors")
            logger.warning("TEST: Streaming had errors")
            return False
        
        if token_count == 0:
            print("\n⚠ No tokens were streamed")
            logger.warning("TEST: No tokens streamed")
            return False
        
        print("\n✅ Streaming working correctly!")
        logger.info("TEST: Streaming successful")
        return True
        
    except Exception as e:
        print(f"\n❌ Streaming test failed: {e}")
        logger.error(f"TEST: Streaming error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print_section("HRBOT BACKEND PIPELINE TEST SUITE")
    
    results = []
    
    # Test 1: Model initialization
    results.append(("Model Initialization", test_model_initialization()))
    
    # Test 2: RAG pipeline
    results.append(("RAG Pipeline", test_rag_pipeline()))
    
    # Test 3: Response generation
    results.append(("Response Generation", test_response_generation()))
    
    # Test 4: Streaming
    results.append(("Streaming", await test_streaming()))
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend pipeline is working correctly.")
        logger.info("TEST: All tests passed")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See logs for details.")
        logger.warning(f"TEST: {total - passed} test(s) failed")
    
    print("\nLogs saved to: data/logs/agent.log")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
