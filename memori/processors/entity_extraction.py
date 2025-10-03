"""
Entity Extraction Service
Extracts entities from memory content to populate the memory graph
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import openai
from loguru import logger
from pydantic import BaseModel, Field

from memori.database.models import MemoryEntity
from memori.utils.pydantic_models import EntityType


class ExtractedEntityWithMetadata(BaseModel):
    """Structured entity extraction output"""

    entity_type: str = Field(description="Type of entity (person, technology, topic, etc.)")
    entity_value: str = Field(description="The actual entity value/name")
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Relevance to the memory (0-1)"
    )
    context: Optional[str] = Field(
        default=None, description="Brief context about this entity in the memory"
    )


class EntityExtractionResult(BaseModel):
    """Complete entity extraction result"""

    entities: List[ExtractedEntityWithMetadata] = Field(
        description="List of extracted entities"
    )
    extraction_confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.8,
        description="Overall confidence in extraction",
    )


class EntityExtractionService:
    """
    Service for extracting entities from memory content using LLM structured outputs
    """

    EXTRACTION_PROMPT = """You are an entity extraction specialist. Extract all relevant entities from the given memory content.

**ENTITY TYPES TO EXTRACT:**

1. **person** - People, names, authors, developers, team members
   Examples: "John Smith", "Alice", "the team lead"

2. **technology** - Technologies, tools, libraries, frameworks, languages
   Examples: "Python", "Docker", "JWT", "PostgreSQL", "FastAPI"

3. **topic** - Topics, concepts, subjects, themes
   Examples: "authentication", "machine learning", "database design"

4. **skill** - Skills, abilities, competencies, expertise areas
   Examples: "API development", "code review", "debugging"

5. **project** - Projects, repositories, applications, systems
   Examples: "user-dashboard", "payment-service", "mobile-app"

6. **keyword** - Important keywords, terms, acronyms
   Examples: "API key", "rate limiting", "CI/CD"

**EXTRACTION GUIDELINES:**
- Extract entities that are central to the memory's meaning
- Assign relevance scores based on importance to the memory
- Provide brief context about how the entity appears in the memory
- Normalize entity values (e.g., "jwt" → "JWT", "docker" → "Docker")
- Extract both explicit mentions and implicit references
- Prioritize quality over quantity (5-10 high-quality entities is better than 20 low-quality)

**RELEVANCE SCORING:**
- 1.0: Central to the memory's core meaning
- 0.8: Important supporting entity
- 0.6: Relevant but not critical
- 0.4: Mentioned but peripheral
- 0.2: Barely relevant

Extract entities now."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "gpt-4o-mini",
        batch_size: int = 10,
    ):
        """
        Initialize entity extraction service

        Args:
            client: OpenAI client instance
            model: Model to use for extraction (default: gpt-4o-mini for speed)
            batch_size: Number of memories to process in batch
        """
        self.client = client
        self.model = model
        self.batch_size = batch_size

        # Statistics
        self.stats = {
            "total_extractions": 0,
            "total_entities": 0,
            "avg_entities_per_memory": 0.0,
            "extraction_errors": 0,
        }

        logger.info(f"EntityExtractionService initialized with model={model}")

    def extract_entities(
        self,
        memory_id: str,
        memory_type: str,
        content: str,
        namespace: str = "default",
    ) -> List[MemoryEntity]:
        """
        Extract entities from a single memory

        Args:
            memory_id: Memory identifier
            memory_type: 'short_term' or 'long_term'
            content: Memory content text
            namespace: Memory namespace

        Returns:
            List of MemoryEntity objects ready to insert
        """
        try:
            # Call LLM for extraction (marked as system to avoid context injection loops)
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"{self.EXTRACTION_PROMPT}\n\nExtract entities from this memory:\n\n{content}",
                    },
                ],
                response_format=EntityExtractionResult,
                temperature=0.3,
            )

            if completion.choices[0].message.refusal:
                logger.warning(f"Entity extraction refused for {memory_id}")
                return []

            result: EntityExtractionResult = completion.choices[0].message.parsed

            # Convert to MemoryEntity objects
            entities = []
            for extracted in result.entities:
                entity = MemoryEntity(
                    entity_id=str(uuid.uuid4()),
                    memory_id=memory_id,
                    memory_type=memory_type,
                    entity_type=extracted.entity_type,
                    entity_value=extracted.entity_value,
                    normalized_value=extracted.entity_value.lower().strip(),
                    relevance_score=extracted.relevance_score,
                    namespace=namespace,
                    frequency=1,
                    created_at=datetime.utcnow(),
                    context=extracted.context,
                )
                entities.append(entity)

            # Update stats
            self.stats["total_extractions"] += 1
            self.stats["total_entities"] += len(entities)
            self.stats["avg_entities_per_memory"] = (
                self.stats["total_entities"] / self.stats["total_extractions"]
            )

            logger.debug(
                f"Extracted {len(entities)} entities from memory {memory_id[:8]}..."
            )

            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed for {memory_id}: {e}")
            self.stats["extraction_errors"] += 1
            return []

    def extract_entities_batch(
        self,
        memories: List[Dict[str, Any]],
        namespace: str = "default",
    ) -> Dict[str, List[MemoryEntity]]:
        """
        Extract entities from multiple memories in batch

        Args:
            memories: List of memory dicts with 'memory_id', 'content', 'memory_type'
            namespace: Memory namespace

        Returns:
            Dict mapping memory_id to list of MemoryEntity objects
        """
        results = {}

        for memory in memories:
            memory_id = memory.get("memory_id")
            memory_type = memory.get("memory_type", "long_term")
            content = memory.get("content", "")

            if not memory_id or not content:
                logger.warning(f"Skipping memory with missing data: {memory}")
                continue

            entities = self.extract_entities(
                memory_id=memory_id,
                memory_type=memory_type,
                content=content,
                namespace=namespace,
            )

            results[memory_id] = entities

        logger.info(
            f"Batch extraction: {len(results)} memories, "
            f"{sum(len(e) for e in results.values())} total entities"
        )

        return results

    def save_entities(
        self,
        entities: List[MemoryEntity],
        session: Any,
    ) -> int:
        """
        Save extracted entities to database

        Args:
            entities: List of MemoryEntity objects
            session: SQLAlchemy session

        Returns:
            Number of entities saved
        """
        if not entities:
            return 0

        try:
            # Check for duplicates and merge
            existing_map = {}
            for entity in entities:
                # Query existing entity for this memory + entity value
                existing = (
                    session.query(MemoryEntity)
                    .filter(
                        MemoryEntity.memory_id == entity.memory_id,
                        MemoryEntity.normalized_value == entity.normalized_value,
                    )
                    .first()
                )

                if existing:
                    # Update frequency and relevance
                    existing.frequency += 1
                    existing.relevance_score = max(
                        existing.relevance_score, entity.relevance_score
                    )
                    existing_map[entity.entity_id] = existing
                else:
                    # Add new entity
                    session.add(entity)

            session.commit()
            logger.debug(f"Saved {len(entities)} entities to database")
            return len(entities)

        except Exception as e:
            logger.error(f"Failed to save entities: {e}")
            session.rollback()
            return 0

    def backfill_entities(
        self,
        db_manager: Any,
        namespace: str = "default",
        batch_size: int = 50,
        limit: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Backfill entities for existing memories that don't have entities

        Args:
            db_manager: Database manager with session
            namespace: Memory namespace
            batch_size: Memories to process per batch
            limit: Optional limit on total memories to process

        Returns:
            Stats dict with counts
        """
        logger.info(f"Starting entity backfill for namespace={namespace}")

        stats = {
            "memories_processed": 0,
            "entities_created": 0,
            "errors": 0,
        }

        with db_manager.get_session() as session:
            # Find memories without entities
            query = """
                SELECT m.memory_id, m.searchable_content as content, 'long_term' as memory_type
                FROM long_term_memory m
                LEFT JOIN memory_entities e ON m.memory_id = e.memory_id
                WHERE m.namespace = :namespace
                  AND e.entity_id IS NULL

                UNION ALL

                SELECT m.memory_id, m.searchable_content as content, 'short_term' as memory_type
                FROM short_term_memory m
                LEFT JOIN memory_entities e ON m.memory_id = e.memory_id
                WHERE m.namespace = :namespace
                  AND e.entity_id IS NULL
            """

            if limit:
                query += f" LIMIT {limit}"

            from sqlalchemy import text

            result = session.execute(text(query), {"namespace": namespace})
            memories_to_process = [
                {"memory_id": row[0], "content": row[1], "memory_type": row[2]}
                for row in result
            ]

            logger.info(f"Found {len(memories_to_process)} memories needing entities")

            # Process in batches
            for i in range(0, len(memories_to_process), batch_size):
                batch = memories_to_process[i : i + batch_size]

                try:
                    # Extract entities for batch
                    batch_results = self.extract_entities_batch(batch, namespace)

                    # Save all entities
                    for memory_id, entities in batch_results.items():
                        saved = self.save_entities(entities, session)
                        stats["entities_created"] += saved
                        stats["memories_processed"] += 1

                    logger.info(
                        f"Backfill progress: {stats['memories_processed']}/{len(memories_to_process)} memories"
                    )

                except Exception as e:
                    logger.error(f"Batch backfill failed: {e}")
                    stats["errors"] += 1
                    continue

        logger.success(
            f"Backfill complete: {stats['memories_processed']} memories, "
            f"{stats['entities_created']} entities created"
        )

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return self.stats.copy()
