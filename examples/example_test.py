#!/usr/bin/env python3
"""
Example script to test RAG system
Run this after starting all services
"""

import json
import time

import httpx

BASE_URL = "http://localhost:8000"
client = httpx.Client()


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_health():
    """Test health check"""
    print_section("Testing Health Check")
    response = client.get(f"{BASE_URL}/health")
    result = response.json()
    print(json.dumps(result, indent=2))
    return result["status"] == "healthy"


def test_add_document():
    """Test adding a document"""
    print_section("Adding Sample Document")

    doc = {
        "title": "Introduction to Machine Learning",
        "content": """
Machine learning is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed. It focuses on the
development of algorithms and models.

Types of Machine Learning:
1. Supervised Learning: The model learns from labeled data. Examples include classification
   and regression tasks. Popular algorithms include decision trees, random forests, and
   support vector machines.

2. Unsupervised Learning: The model learns patterns from unlabeled data. Examples include
   clustering and dimensionality reduction. K-means and hierarchical clustering are common.

3. Reinforcement Learning: The model learns by interacting with an environment through
   trial and error, receiving rewards or penalties for actions.

Applications:
- Computer Vision: Image recognition, object detection, medical imaging
- Natural Language Processing: Text classification, machine translation, sentiment analysis
- Recommendation Systems: Product recommendations, content personalization
- Fraud Detection: Identifying suspicious transactions in financial systems
- Autonomous Vehicles: Perception, planning, and control systems

Neural Networks:
Deep learning uses artificial neural networks with multiple layers to learn complex patterns.
Convolutional neural networks (CNNs) are effective for image tasks, while recurrent neural
networks (RNNs) work well for sequential data.
        """,
        "metadata": {"source": "knowledge_base", "category": "AI"},
    }

    response = client.post(f"{BASE_URL}/add-document", json=doc)
    result = response.json()
    print(json.dumps(result, indent=2))
    return result.get("doc_id")


def test_search(query: str):
    """Test hybrid search"""
    print_section(f"Searching: '{query}'")

    request = {"query": query, "top_k": 3, "use_hybrid": True}

    response = client.post(f"{BASE_URL}/search", json=request)
    result = response.json()

    print(f"Query: {result['query']}")
    print(f"Results found: {result['total_results']}\n")

    for i, res in enumerate(result["results"], 1):
        print(f"Result {i}:")
        print(f"  Score: {res['score']:.4f}")
        print(f"  Source: {res['source']}")
        print(f"  Content: {res['content'][:100]}...")
        print()


def test_rag_query(query: str):
    """Test RAG query (search + generate)"""
    print_section(f"RAG Query: '{query}'")

    request = {"query": query, "top_k": 3, "use_hybrid": True}

    response = client.post(f"{BASE_URL}/query", json=request, timeout=120.0)
    result = response.json()

    print(f"Query: {result['query']}")
    print("\nGenerated Answer:")
    print(result["answer"])
    print(f"\nGeneration Time: {result['generation_time']:.2f}s")
    print(f"\nSource Documents: {len(result['sources'])}")


def main():
    """Run all tests"""
    print("🚀 RAG Hybrid Search System - Test Suite\n")

    # Test health
    if not test_health():
        print("\n❌ System is not healthy. Please ensure:")
        print("   1. Ollama is running: ollama serve")
        print("   2. Qdrant is running: docker-compose up -d")
        print("   3. FastAPI server is running: python -m uvicorn app.main:app --reload")
        return

    print("\n✅ System is healthy!")

    # Add document
    test_add_document()

    # Wait a moment for indexing
    time.sleep(2)

    # Test search and RAG
    test_search("What types of machine learning exist?")
    test_search("neural networks deep learning")

    test_rag_query("What are the applications of machine learning?")
    test_rag_query("How does supervised learning work?")

    print("\n" + "=" * 60)
    print("  ✅ All tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
