import os
import uuid
from typing import Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from .embeddings import embed_single, embed_text_sync, EMBED_DIMENSIONS
from .models import Pattern, PatternMetadata, ReasoningTrace

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "patterns"

# Namespace for generating deterministic UUIDs from pattern_ids
PATTERN_UUID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def pattern_id_to_uuid(pattern_id: str) -> str:
    """Convert string pattern_id to a deterministic UUID."""
    return str(uuid.uuid5(PATTERN_UUID_NAMESPACE, pattern_id))


class KnowledgeGraph:
    """
    Stores problem-solution patterns and enables semantic search.
    """

    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBED_DIMENSIONS,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {COLLECTION_NAME}")

    def _pattern_to_searchable_text(self, pattern: Pattern) -> str:
        """Convert pattern to text for embedding."""
        return f"{pattern.problem_class}\n{pattern.problem_signature}\n{' '.join(pattern.domain_tags)}"

    async def add_pattern(self, pattern: Pattern) -> str:
        """Add a pattern to the knowledge graph."""
        searchable_text = self._pattern_to_searchable_text(pattern)
        vector = await embed_single(searchable_text)
        point_id = pattern_id_to_uuid(pattern.pattern_id)

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=pattern.model_dump(mode="json")
        )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

        return pattern.pattern_id

    def add_pattern_sync(self, pattern: Pattern) -> str:
        """Synchronous version for batch indexing."""
        searchable_text = self._pattern_to_searchable_text(pattern)
        vectors = embed_text_sync(searchable_text)
        vector = vectors[0]
        point_id = pattern_id_to_uuid(pattern.pattern_id)

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=pattern.model_dump(mode="json")
        )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

        return pattern.pattern_id

    async def search(
        self,
        query: str,
        domain_filter: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> list[dict]:
        """
        Search for patterns matching a query.

        Args:
            query: Problem description to search for
            domain_filter: Optional domain tag to filter by
            limit: Max results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of matching patterns with scores
        """
        query_vector = await embed_single(query)

        search_filter = None
        if domain_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="domain_tags",
                        match=MatchAny(any=[domain_filter])
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        )

        return [
            {
                "pattern": point.payload,
                "score": point.score,
                "pattern_id": point.payload.get("pattern_id", point.id)
            }
            for point in results.points
        ]

    async def get_pattern(self, pattern_id: str) -> Optional[dict]:
        """Get a specific pattern by ID."""
        point_id = pattern_id_to_uuid(pattern_id)
        results = self.client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True
        )

        if results:
            return results[0].payload
        return None

    async def update_pattern_stats(
        self,
        pattern_id: str,
        success: bool,
        refinement: Optional[str] = None
    ):
        """Update pattern statistics after it's been used."""
        pattern_data = await self.get_pattern(pattern_id)
        if not pattern_data:
            return

        # Update stats
        pattern_data["metadata"]["times_applied"] += 1
        if success:
            # Recalculate success rate
            total = pattern_data["metadata"]["times_applied"]
            current_rate = pattern_data["metadata"]["success_rate"]
            new_rate = ((current_rate * (total - 1)) + 1.0) / total
            pattern_data["metadata"]["success_rate"] = new_rate

        if refinement:
            pattern_data["refinements"].append(refinement)

        # Re-embed and update (in case description changed)
        pattern = Pattern(**pattern_data)
        await self.add_pattern(pattern)

    def get_stats(self) -> dict:
        """Get knowledge graph statistics."""
        collection_info = self.client.get_collection(COLLECTION_NAME)
        return {
            "total_patterns": collection_info.points_count,
            "indexed_vectors": collection_info.indexed_vectors_count,
            "status": str(collection_info.status)
        }

    def delete_all(self):
        """Delete all patterns (for testing)."""
        self.client.delete_collection(COLLECTION_NAME)
        self._ensure_collection()


# Singleton instance
_knowledge_graph: Optional[KnowledgeGraph] = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Get or create the knowledge graph instance."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph
