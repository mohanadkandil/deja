"""
Test script to verify the foundation works.
Run: uv run python test_foundation.py
"""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

def test_env():
    """Check required env vars are set."""
    print("\n1. Checking environment variables...")

    required = ["MISTRAL_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]

    if missing:
        print(f"   ❌ Missing: {missing}")
        print("   → Copy .env.example to .env and fill in your keys")
        return False

    print("   ✅ MISTRAL_API_KEY is set")
    return True


async def test_embeddings():
    """Test Mistral embeddings API."""
    print("\n2. Testing Mistral Embeddings...")

    from backend.embeddings import embed_single

    try:
        vector = await embed_single("webhook dropping events under load")
        print(f"   ✅ Got embedding vector with {len(vector)} dimensions")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_qdrant():
    """Test Qdrant connection."""
    print("\n3. Testing Qdrant connection...")

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print(f"   ✅ Connected to Qdrant. Collections: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   → Make sure Qdrant is running: docker-compose up -d")
        return False


async def test_knowledge_graph():
    """Test knowledge graph with a sample pattern."""
    print("\n4. Testing Knowledge Graph...")

    from backend.knowledge_graph import get_knowledge_graph
    from backend.models import Pattern, PatternMetadata, ReasoningTrace

    try:
        kg = get_knowledge_graph()

        # Create a test pattern
        pattern = Pattern(
            pattern_id="TEST-001",
            problem_class="Webhook event loss under load",
            problem_signature="HTTP endpoint receiving burst traffic drops events due to synchronous processing",
            domain_tags=["webhooks", "concurrency", "api"],
            reasoning_trace=ReasoningTrace(
                failed_approaches=[],
                key_insight="Decouple reception from processing using a message queue"
            ),
            solution_template="1. Write raw payload to queue immediately\n2. Return 200 within 50ms\n3. Process async from queue",
            metadata=PatternMetadata(
                source="test",
                verification_status="verified",
                success_rate=0.95,
                times_applied=0,
                estimated_token_savings=12000,
                difficulty="medium"
            )
        )

        # Add pattern
        await kg.add_pattern(pattern)
        print("   ✅ Added test pattern")

        # Search for it
        results = await kg.search("webhook dropping events", limit=3)

        if results and results[0]["pattern_id"] == "TEST-001":
            print(f"   ✅ Search works! Found pattern with score: {results[0]['score']:.3f}")
        else:
            print("   ⚠️  Search returned results but didn't find test pattern")

        # Get stats
        stats = kg.get_stats()
        print(f"   ✅ Knowledge graph stats: {stats}")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("=" * 50)
    print("ECHO Foundation Test")
    print("=" * 50)

    results = []

    # Test 1: Environment
    results.append(("Environment", test_env()))

    if not results[-1][1]:
        print("\n⛔ Fix environment variables first!")
        return

    # Test 2: Qdrant
    results.append(("Qdrant", test_qdrant()))

    if not results[-1][1]:
        print("\n⛔ Start Qdrant first: docker-compose up -d")
        return

    # Test 3: Embeddings
    results.append(("Embeddings", await test_embeddings()))

    # Test 4: Knowledge Graph
    results.append(("Knowledge Graph", await test_knowledge_graph()))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed! Foundation is solid.")
    else:
        print("\n⚠️  Some tests failed. Fix issues above.")


if __name__ == "__main__":
    asyncio.run(main())
