"""
Relationship Detection Service
Automatically detects and creates relationships between memories
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from memori.database.models import MemoryEntity, MemoryRelationshipDB
from memori.utils.pydantic_models import RelationshipType


class RelationshipDetectionService:
    """
    Service for detecting relationships between memories
    Uses entity overlap, semantic similarity, and heuristics
    """

    def __init__(self, db_manager: Any, threshold_config: Optional[Dict[str, float]] = None):
        """
        Initialize relationship detection service

        Args:
            db_manager: Database manager instance
            threshold_config: Optional thresholds for relationship creation
        """
        self.db_manager = db_manager

        # Configurable thresholds
        self.thresholds = threshold_config or {
            "min_entity_overlap": 1,  # Minimum shared entities (lowered for small databases)
            "min_strength": 0.2,  # Minimum relationship strength (lowered to be more permissive)
            "entity_overlap_weight": 0.6,  # Weight for entity overlap
            "temporal_proximity_weight": 0.2,  # Weight for time proximity
            "category_match_weight": 0.2,  # Weight for category match
        }

        # Statistics
        self.stats = {
            "relationships_created": 0,
            "relationships_updated": 0,
            "memories_analyzed": 0,
        }

        logger.info("RelationshipDetectionService initialized")

    def detect_relationships_for_memory(
        self,
        memory_id: str,
        memory_type: str,
        namespace: str = "default",
        max_candidates: int = 50,
    ) -> List[MemoryRelationshipDB]:
        """
        Detect relationships for a newly added memory

        Args:
            memory_id: Memory identifier
            memory_type: 'short_term' or 'long_term'
            namespace: Memory namespace
            max_candidates: Maximum candidate memories to compare

        Returns:
            List of detected relationships
        """
        logger.debug(f"Detecting relationships for memory {memory_id[:8]}...")

        relationships = []

        with self.db_manager.get_session() as session:
            # Step 1: Get entities for this memory
            source_entities = (
                session.query(MemoryEntity)
                .filter(
                    MemoryEntity.memory_id == memory_id,
                    MemoryEntity.namespace == namespace,
                )
                .all()
            )

            if not source_entities:
                logger.debug(f"No entities found for memory {memory_id}, skipping")
                return []

            source_entity_values = {e.normalized_value for e in source_entities}

            # Step 2: Find candidate memories with overlapping entities
            candidates = self._find_candidate_memories(
                session,
                source_entity_values,
                memory_id,
                namespace,
                max_candidates,
            )

            logger.debug(f"Found {len(candidates)} candidate memories for comparison")

            # Step 3: Calculate relationship strength for each candidate
            for candidate in candidates:
                relationship = self._calculate_relationship(
                    source_memory_id=memory_id,
                    source_memory_type=memory_type,
                    target_memory_id=candidate["memory_id"],
                    target_memory_type=candidate["memory_type"],
                    shared_entities=candidate["shared_entities"],
                    namespace=namespace,
                )

                if relationship and relationship.strength >= self.thresholds["min_strength"]:
                    relationships.append(relationship)

            # Step 4: Save relationships
            if relationships:
                self._save_relationships(session, relationships)

        self.stats["memories_analyzed"] += 1
        self.stats["relationships_created"] += len(relationships)

        logger.info(
            f"Created {len(relationships)} relationships for memory {memory_id[:8]}..."
        )

        return relationships

    def _find_candidate_memories(
        self,
        session: Any,
        source_entities: set,
        exclude_memory_id: str,
        namespace: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Find memories with overlapping entities"""

        from sqlalchemy import func, text

        # Build entity filter
        entity_list = list(source_entities)
        entity_placeholders = ",".join([f":entity_{i}" for i in range(len(entity_list))])

        # Some dialects (PostgreSQL) use string_agg instead of GROUP_CONCAT
        dialect = getattr(session.bind.dialect, "name", "sqlite")
        if dialect == "postgresql":
            aggregate_expr = "string_agg(e.normalized_value, ',' ORDER BY e.normalized_value)"
        else:
            aggregate_expr = "GROUP_CONCAT(e.normalized_value, ',')"

        count_expr = "COUNT(DISTINCT e.normalized_value)"

        query = f"""
        SELECT
            e.memory_id,
            e.memory_type,
            {aggregate_expr} as shared_entities,
            {count_expr} as overlap_count
        FROM memory_entities e
        WHERE e.namespace = :namespace
          AND e.memory_id != :exclude_memory_id
          AND e.normalized_value IN ({entity_placeholders})
        GROUP BY e.memory_id, e.memory_type
        HAVING {count_expr} >= :min_overlap
        ORDER BY overlap_count DESC
        LIMIT :limit_val
        """

        # Build params dict with named parameters
        params = {
            "namespace": namespace,
            "exclude_memory_id": exclude_memory_id,
            "min_overlap": self.thresholds["min_entity_overlap"],
            "limit_val": limit,
        }

        # Add entity parameters
        for i, entity in enumerate(entity_list):
            params[f"entity_{i}"] = entity

        result = session.execute(text(query), params)

        candidates = []
        for row in result:
            candidates.append(
                {
                    "memory_id": row[0],
                    "memory_type": row[1],
                    "shared_entities": row[2].split(",") if row[2] else [],
                    "overlap_count": row[3],
                }
            )

        return candidates

    def _calculate_relationship(
        self,
        source_memory_id: str,
        source_memory_type: str,
        target_memory_id: str,
        target_memory_type: str,
        shared_entities: List[str],
        namespace: str,
    ) -> Optional[MemoryRelationshipDB]:
        """
        Calculate relationship strength and type between two memories

        Uses a composite scoring approach:
        - Entity overlap (primary signal)
        - Temporal proximity (memories created around same time)
        - Category match (memories in same category)
        """

        # Calculate entity overlap score (0-1). Cap at 3 shared entities so that
        # even modest overlap produces a meaningful signal.
        overlap_count = len(shared_entities)
        entity_overlap_score = min(1.0, overlap_count / 3.0)

        # Baseline strength ensures that a single shared entity still produces a
        # traversable edge once graph thresholds are applied. Remaining weight is
        # driven by actual overlap so denser connections still rank higher.
        base_strength = 0.2
        overlap_component = (
            entity_overlap_score * self.thresholds["entity_overlap_weight"]
        )
        strength = min(1.0, base_strength + overlap_component)

        # TODO: incorporate temporal and category proximity when available. For
        # now the baseline + overlap component keeps edges discoverable without
        # inflating low-value links.

        # Determine relationship type based on shared entities
        relationship_type = self._infer_relationship_type(shared_entities, overlap_count)

        # Create relationship
        relationship = MemoryRelationshipDB(
            relationship_id=str(uuid.uuid4()),
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            source_memory_type=source_memory_type,
            target_memory_type=target_memory_type,
            relationship_type=relationship_type,
            strength=strength,
            bidirectional=True,
            namespace=namespace,
            created_at=datetime.utcnow(),
            shared_entity_count=overlap_count,
            reasoning=f"Shared {overlap_count} entities: {', '.join(shared_entities[:3])}",
        )

        return relationship

    def _infer_relationship_type(
        self, shared_entities: List[str], overlap_count: int
    ) -> str:
        """
        Infer relationship type based on shared entities and context

        For now, uses simple heuristics:
        - High overlap (4+) → semantic_similarity
        - Medium overlap (2-3) → related_entity
        - Specific patterns → other types (future enhancement)
        """

        if overlap_count >= 4:
            return RelationshipType.SEMANTIC_SIMILARITY.value

        # Default to related_entity for shared entities
        return RelationshipType.RELATED_ENTITY.value

    def _save_relationships(
        self, session: Any, relationships: List[MemoryRelationshipDB]
    ) -> int:
        """Save relationships to database, handling duplicates"""

        saved_count = 0

        for rel in relationships:
            # Check if relationship already exists (bidirectional check)
            existing = (
                session.query(MemoryRelationshipDB)
                .filter(
                    (
                        (MemoryRelationshipDB.source_memory_id == rel.source_memory_id)
                        & (MemoryRelationshipDB.target_memory_id == rel.target_memory_id)
                    )
                    | (
                        (MemoryRelationshipDB.source_memory_id == rel.target_memory_id)
                        & (MemoryRelationshipDB.target_memory_id == rel.source_memory_id)
                    )
                )
                .first()
            )

            if existing:
                # Update existing relationship
                existing.strength = max(existing.strength, rel.strength)
                existing.shared_entity_count = max(
                    existing.shared_entity_count, rel.shared_entity_count
                )
                existing.last_strengthened = datetime.utcnow()
                self.stats["relationships_updated"] += 1
            else:
                # Create new relationship
                session.add(rel)
                saved_count += 1

        session.commit()
        return saved_count

    def backfill_relationships(
        self,
        namespace: str = "default",
        batch_size: int = 100,
        limit: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Backfill relationships for all existing memories

        Args:
            namespace: Memory namespace
            batch_size: Memories to process per batch
            limit: Optional limit on total memories

        Returns:
            Stats dict with counts
        """
        logger.info(f"Starting relationship backfill for namespace={namespace}")

        stats = {
            "memories_processed": 0,
            "relationships_created": 0,
        }

        with self.db_manager.get_session() as session:
            # Get all memories with entities
            from sqlalchemy import text

            query = """
                SELECT DISTINCT e.memory_id, e.memory_type
                FROM memory_entities e
                WHERE e.namespace = :namespace
            """

            if limit:
                query += f" LIMIT {limit}"

            result = session.execute(text(query), {"namespace": namespace})
            memories = [{"memory_id": row[0], "memory_type": row[1]} for row in result]

            logger.info(f"Found {len(memories)} memories to process")

            # Process each memory
            for i, memory in enumerate(memories):
                try:
                    relationships = self.detect_relationships_for_memory(
                        memory_id=memory["memory_id"],
                        memory_type=memory["memory_type"],
                        namespace=namespace,
                    )

                    stats["relationships_created"] += len(relationships)
                    stats["memories_processed"] += 1

                    if (i + 1) % 10 == 0:
                        logger.info(
                            f"Backfill progress: {i+1}/{len(memories)} memories processed"
                        )

                except Exception as e:
                    logger.error(f"Failed to process memory {memory['memory_id']}: {e}")
                    continue

        logger.success(
            f"Backfill complete: {stats['memories_processed']} memories, "
            f"{stats['relationships_created']} relationships created"
        )

        return stats

    def strengthen_relationship(
        self, relationship_id: str, strength_increase: float = 0.1
    ) -> bool:
        """
        Strengthen an existing relationship (e.g., when traversed/accessed)

        Args:
            relationship_id: Relationship identifier
            strength_increase: Amount to increase strength by

        Returns:
            True if successful
        """
        with self.db_manager.get_session() as session:
            rel = (
                session.query(MemoryRelationshipDB)
                .filter(MemoryRelationshipDB.relationship_id == relationship_id)
                .first()
            )

            if rel:
                rel.strength = min(1.0, rel.strength + strength_increase)
                rel.access_count += 1
                rel.last_strengthened = datetime.utcnow()
                session.commit()
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        return self.stats.copy()
