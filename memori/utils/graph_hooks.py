"""
Graph Building Hooks
Automatic entity extraction and relationship detection when memories are stored
"""

from typing import Any, Dict, Optional

from loguru import logger


class MemoryStorageHook:
    """
    Hook system for automatic graph building when memories are stored

    Usage:
        hook = MemoryStorageHook(
            entity_extractor=mem.entity_extractor,
            relationship_detector=mem.relationship_detector,
            enabled=True
        )

        # After storing a memory
        hook.process_memory(
            memory_id=memory_id,
            memory_type="long_term",
            content=content,
            namespace=namespace
        )
    """

    def __init__(
        self,
        entity_extractor: Any = None,
        relationship_detector: Any = None,
        enabled: bool = True,
        async_processing: bool = False,
    ):
        """
        Initialize storage hook

        Args:
            entity_extractor: EntityExtractionService instance
            relationship_detector: RelationshipDetectionService instance
            enabled: Enable automatic graph building
            async_processing: Process in background (future enhancement)
        """
        self.entity_extractor = entity_extractor
        self.relationship_detector = relationship_detector
        self.enabled = enabled
        self.async_processing = async_processing

        # Statistics
        self.stats = {
            "memories_processed": 0,
            "entities_extracted": 0,
            "relationships_created": 0,
            "errors": 0,
        }

        logger.info(
            f"MemoryStorageHook initialized (enabled={enabled}, "
            f"async={async_processing})"
        )

    def process_memory(
        self,
        memory_id: str,
        memory_type: str,
        content: str,
        namespace: str = "default",
        db_session: Any = None,
    ) -> Dict[str, int]:
        """
        Process a newly stored memory to build graph

        Args:
            memory_id: Memory identifier
            memory_type: 'short_term' or 'long_term'
            content: Memory content text
            namespace: Memory namespace
            db_session: Optional database session for batch operations

        Returns:
            Stats dict with counts
        """
        if not self.enabled:
            return {"entities": 0, "relationships": 0}

        if not self.entity_extractor or not self.relationship_detector:
            logger.warning("Graph components not initialized, skipping graph building")
            return {"entities": 0, "relationships": 0}

        stats = {"entities": 0, "relationships": 0}

        try:
            # Step 1: Extract entities
            logger.debug(f"Extracting entities for memory {memory_id[:8]}...")
            entities = self.entity_extractor.extract_entities(
                memory_id=memory_id,
                memory_type=memory_type,
                content=content,
                namespace=namespace,
            )

            # Step 2: Save entities
            if entities and db_session:
                saved = self.entity_extractor.save_entities(entities, db_session)
                stats["entities"] = saved
                self.stats["entities_extracted"] += saved
            elif entities:
                # Need to create session
                logger.warning("No db_session provided, entities not saved")

            # Step 3: Detect relationships
            logger.debug(f"Detecting relationships for memory {memory_id[:8]}...")
            relationships = self.relationship_detector.detect_relationships_for_memory(
                memory_id=memory_id,
                memory_type=memory_type,
                namespace=namespace,
                max_candidates=50,
            )

            stats["relationships"] = len(relationships)
            self.stats["relationships_created"] += len(relationships)
            self.stats["memories_processed"] += 1

            logger.info(
                f"Graph built for {memory_id[:8]}: "
                f"{stats['entities']} entities, {stats['relationships']} relationships"
            )

            return stats

        except Exception as e:
            logger.error(f"Graph building failed for {memory_id}: {e}")
            self.stats["errors"] += 1
            return {"entities": 0, "relationships": 0}

    def process_memory_batch(
        self,
        memories: list[Dict[str, Any]],
        namespace: str = "default",
    ) -> Dict[str, int]:
        """
        Process multiple memories in batch

        Args:
            memories: List of memory dicts with 'memory_id', 'content', 'memory_type'
            namespace: Memory namespace

        Returns:
            Aggregate stats
        """
        if not self.enabled:
            return {"total_entities": 0, "total_relationships": 0}

        total_stats = {"total_entities": 0, "total_relationships": 0}

        for memory in memories:
            stats = self.process_memory(
                memory_id=memory.get("memory_id"),
                memory_type=memory.get("memory_type", "long_term"),
                content=memory.get("content", ""),
                namespace=namespace,
            )

            total_stats["total_entities"] += stats.get("entities", 0)
            total_stats["total_relationships"] += stats.get("relationships", 0)

        logger.info(
            f"Batch processing complete: {len(memories)} memories, "
            f"{total_stats['total_entities']} entities, "
            f"{total_stats['total_relationships']} relationships"
        )

        return total_stats

    def get_stats(self) -> Dict[str, int]:
        """Get hook statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "memories_processed": 0,
            "entities_extracted": 0,
            "relationships_created": 0,
            "errors": 0,
        }

    def enable(self):
        """Enable automatic graph building"""
        self.enabled = True
        logger.info("MemoryStorageHook enabled")

    def disable(self):
        """Disable automatic graph building"""
        self.enabled = False
        logger.info("MemoryStorageHook disabled")


def create_storage_hook(memori_instance: Any, enabled: bool = True) -> MemoryStorageHook:
    """
    Create a storage hook from Memori instance

    Args:
        memori_instance: Memori class instance
        enabled: Enable automatic processing

    Returns:
        Configured MemoryStorageHook

    Example:
        hook = create_storage_hook(mem)

        # Use in storage flow
        memory_id = mem.store_memory(content)
        hook.process_memory(memory_id, "long_term", content, namespace)
    """
    hook = MemoryStorageHook(
        entity_extractor=getattr(memori_instance, "entity_extractor", None),
        relationship_detector=getattr(memori_instance, "relationship_detector", None),
        enabled=enabled,
    )

    return hook
