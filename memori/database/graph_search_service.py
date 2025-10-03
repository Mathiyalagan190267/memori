"""
Graph-Based Memory Search Service
Implements 7 search strategies with graph expansion and composite scoring
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from memori.database.graph_queries import get_query_builder
from memori.utils.pydantic_models import (
    ExpansionStrategy,
    GraphExpansionConfig,
    GraphSearchResult,
    GraphTraversalPath,
    RelationshipType,
    SearchStrategy,
    ScoringWeights,
)


class GraphSearchService:
    """
    Core service for graph-based memory search
    Supports 7 search strategies with configurable scoring
    """

    def __init__(self, database_manager):
        """
        Initialize graph search service

        Args:
            database_manager: Database manager with engine/session
        """
        self.db_manager = database_manager
        self.dialect = database_manager.engine.dialect.name
        self.query_builder = get_query_builder(self.dialect)

        # Performance statistics
        self.stats = {
            "total_searches": 0,
            "strategy_usage": {},
            "avg_response_time_ms": 0,
        }

        logger.info(f"GraphSearchService initialized for {self.dialect}")

    def search(
        self,
        query_text: str,
        strategy: SearchStrategy,
        namespace: str = "default",
        entities: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        graph_expansion: Optional[GraphExpansionConfig] = None,
        scoring_weights: Optional[ScoringWeights] = None,
        max_results: int = 10,
    ) -> List[GraphSearchResult]:
        """
        Main search entry point with strategy routing

        Args:
            query_text: Search query string
            strategy: Search strategy to use
            namespace: Memory namespace
            entities: Optional entity filters
            categories: Optional category filters
            graph_expansion: Graph expansion configuration
            scoring_weights: Scoring weights
            max_results: Maximum results to return

        Returns:
            List of GraphSearchResult objects
        """
        start_time = datetime.now()
        self.stats["total_searches"] += 1
        self.stats["strategy_usage"][strategy] = (
            self.stats["strategy_usage"].get(strategy, 0) + 1
        )

        logger.debug(
            f"Graph search: strategy={strategy}, namespace={namespace}, "
            f"entities={entities}, max_results={max_results}"
        )

        try:
            # Route to appropriate strategy
            if strategy == SearchStrategy.TEXT_ONLY:
                results = self._text_only_search(
                    query_text, namespace, categories, max_results
                )

            elif strategy == SearchStrategy.ENTITY_FIRST:
                results = self.entity_first_search(
                    entities or [],
                    namespace,
                    categories,
                    max_results,
                )

            elif strategy == SearchStrategy.GRAPH_EXPANSION_1HOP:
                results = self.search_with_expansion(
                    query_text,
                    entities or [],
                    categories or [],
                    namespace,
                    expand_hops=1,
                    min_strength=graph_expansion.min_relationship_strength
                    if graph_expansion
                    else 0.2,
                    limit=max_results,
                )

            elif strategy == SearchStrategy.GRAPH_EXPANSION_2HOP:
                results = self.search_with_expansion(
                    query_text,
                    entities or [],
                    categories or [],
                    namespace,
                    expand_hops=2,
                    min_strength=graph_expansion.min_relationship_strength
                    if graph_expansion
                    else 0.2,
                    limit=max_results,
                )

            elif strategy == SearchStrategy.GRAPH_WALK_CONTEXTUAL:
                results = self.graph_walk(
                    entities or [],
                    namespace,
                    max_depth=3,
                    min_strength=graph_expansion.min_relationship_strength
                    if graph_expansion
                    else 0.2,
                    limit=max_results,
                )

            elif strategy == SearchStrategy.ENTITY_CLUSTER_DISCOVERY:
                results = self.entity_cluster_discovery(
                    entities or [],
                    namespace,
                    min_shared=2,
                    limit=max_results,
                )

            elif strategy == SearchStrategy.CATEGORY_FOCUSED_GRAPH:
                results = self.category_focused_graph_search(
                    query_text,
                    categories or [],
                    namespace,
                    expand_hops=1,
                    limit=max_results,
                )

            else:
                logger.warning(f"Unknown strategy: {strategy}, falling back to TEXT_ONLY")
                results = self._text_only_search(
                    query_text, namespace, categories, max_results
                )

            # Apply composite scoring
            if scoring_weights:
                results = self._apply_composite_scoring(results, scoring_weights)

            # Sort by composite score
            results = sorted(results, key=lambda x: x.composite_score, reverse=True)[
                :max_results
            ]

            # Calculate response time
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.stats["avg_response_time_ms"] = (
                self.stats["avg_response_time_ms"] * 0.9 + elapsed_ms * 0.1
            )  # Moving average

            logger.info(
                f"Search completed: strategy={strategy}, results={len(results)}, "
                f"time={elapsed_ms:.1f}ms"
            )

            return results

        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            raise

    def search_with_expansion(
        self,
        query_text: str,
        entities: List[str],
        categories: List[str],
        namespace: str,
        expand_hops: int = 1,
        min_strength: float = 0.5,
        limit: int = 10,
    ) -> List[GraphSearchResult]:
        """
        Search with graph expansion from seed memories

        Steps:
        1. Find seed memories (text + entity search)
        2. Expand via graph relationships
        3. Combine and deduplicate results
        4. Calculate composite scores
        5. Return top results with graph metadata
        """
        # Step 1: Find seed memories
        seed_memories = self._find_seed_memories(
            query_text, entities, categories, namespace, limit=20
        )

        if not seed_memories:
            logger.debug(
                "No seed memories found via entities, falling back to text search for seeds"
            )
            # Fallback: Use text-based search to find seed memories
            seed_memories = self._find_text_based_seeds(
                query_text, namespace, categories, limit=10
            )

        if not seed_memories:
            logger.debug(
                "No seed memories found via text search either, graph search cannot proceed"
            )
            return []

        seed_memory_ids = [m["memory_id"] for m in seed_memories]
        logger.debug(f"Starting graph expansion from {len(seed_memory_ids)} seed memories")

        # Step 2: Expand via graph
        expanded_memories = self._expand_via_graph(
            seed_memory_ids,
            hop_distance=expand_hops,
            min_strength=min_strength,
            namespace=namespace,
            limit_per_hop=limit,
        )

        # Step 3: Combine seed and expanded memories
        all_memories = seed_memories + expanded_memories

        # Step 4: Fetch full memory data and enrich with graph metadata
        results = self._enrich_with_memory_data(all_memories, namespace)

        # Step 5: Generate match reasons
        results = self._generate_match_reasons(results, query_text, entities)

        return results

    def entity_first_search(
        self,
        entities: List[str],
        namespace: str,
        categories: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[GraphSearchResult]:
        """Search by entity tags first, then expand"""
        if not entities:
            return []

        # Build and execute entity search query
        query, params = self.query_builder.build_entity_search_query(
            entities=entities,
            entity_types=None,
            namespace=namespace,
            min_relevance=0.3,
            limit=limit,
        )

        with self.db_manager.get_session() as session:
            raw_results = self._execute_query(session, query, params)

        # Convert to GraphSearchResult
        results = []
        for row in raw_results:
            result = GraphSearchResult(
                memory_id=row["memory_id"],
                content="",  # Will be filled by _enrich_with_memory_data
                summary="",
                entity_overlap_score=min(1.0, row.get("entity_match_count", 0) / len(entities)),
                hop_distance=0,
                shared_entities=(
                    row.get("matched_entities", "").split(", ")
                    if isinstance(row.get("matched_entities"), str)
                    else row.get("matched_entities", [])
                ),
                match_reason=f"Matched {row.get('entity_match_count', 0)} entities",
            )
            results.append(result)

        # Enrich with full memory data
        results = self._enrich_with_memory_data(
            [{"memory_id": r.memory_id, "hop": 0} for r in results],
            namespace,
        )

        return results

    def entity_cluster_discovery(
        self,
        entities: List[str],
        namespace: str,
        min_shared: int = 2,
        limit: int = 50,
    ) -> List[GraphSearchResult]:
        """Find memories that share multiple entities (cluster discovery)"""
        if not entities or len(entities) < min_shared:
            return []

        # Build and execute entity cluster query
        query, params = self.query_builder.build_entity_cluster_query(
            entities=entities,
            namespace=namespace,
            min_shared_entities=min_shared,
            limit=limit,
        )

        with self.db_manager.get_session() as session:
            raw_results = self._execute_query(session, query, params)

        # Convert to results
        results = []
        for row in raw_results:
            shared_entities = (
                row.get("shared_entities", "").split(", ")
                if isinstance(row.get("shared_entities"), str)
                else row.get("shared_entities", [])
            )

            result = GraphSearchResult(
                memory_id=row["memory_id"],
                content="",
                summary="",
                entity_overlap_score=min(
                    1.0, row.get("shared_entity_count", 0) / len(entities)
                ),
                hop_distance=0,
                shared_entities=shared_entities,
                match_reason=f"Shared {row.get('shared_entity_count', 0)} entities: {', '.join(shared_entities[:3])}",
            )
            results.append(result)

        # Enrich with full data
        results = self._enrich_with_memory_data(
            [{"memory_id": r.memory_id, "hop": 0} for r in results],
            namespace,
        )

        return results

    def graph_walk(
        self,
        entities: List[str],
        namespace: str,
        max_depth: int = 3,
        min_strength: float = 0.5,
        limit: int = 20,
    ) -> List[GraphSearchResult]:
        """
        Contextual graph walk - follow relationships from entity-tagged memories

        This is the most powerful strategy for "find everything related to X" queries
        """
        # First find memories with these entities
        seed_results = self.entity_first_search(
            entities, namespace, categories=None, limit=10
        )

        if not seed_results:
            return []

        seed_ids = [r.memory_id for r in seed_results]

        # Perform multi-hop expansion
        expanded = self._expand_via_graph(
            seed_ids,
            hop_distance=max_depth,
            min_strength=min_strength,
            namespace=namespace,
            limit_per_hop=limit,
        )

        # Combine seeds and expanded
        all_results = seed_results + self._enrich_with_memory_data(expanded, namespace)

        return all_results[:limit]

    def category_focused_graph_search(
        self,
        query_text: str,
        categories: List[str],
        namespace: str,
        expand_hops: int = 1,
        limit: int = 50,
    ) -> List[GraphSearchResult]:
        """Search within specific categories, then expand via graph"""

        # Find seed memories in categories
        seeds = self._find_seed_memories(
            query_text, [], categories, namespace, limit=20
        )

        if not seeds:
            return []

        # Expand from seeds
        return self.search_with_expansion(
            query_text=query_text,
            entities=[],
            categories=categories,
            namespace=namespace,
            expand_hops=expand_hops,
            min_strength=0.5,
            limit=limit,
        )

    # ==================== Helper Methods ====================

    def _text_only_search(
        self,
        query_text: str,
        namespace: str,
        categories: Optional[List[str]],
        limit: int,
    ) -> List[GraphSearchResult]:
        """Fallback to traditional text search (no graph)"""
        try:
            from .search_service import SearchService

            # Get a session from db_manager
            with self.db_manager.SessionLocal() as session:
                search_service = SearchService(
                    session=session,
                    database_type=self.dialect
                )

                results = search_service.search_memories(
                    query=query_text,
                    namespace=namespace,
                    limit=limit,
                    category_filter=categories,
                )

            # Convert to GraphSearchResult format
            graph_results = []
            for result in results:
                if isinstance(result, dict):
                    composite = result.get("composite_score")
                    if composite is None:
                        composite = result.get("search_score", 0.5)
                    text_score = result.get("search_score", composite)
                    created_at = result.get("created_at", datetime.now())

                    graph_results.append(
                        GraphSearchResult(
                            memory_id=result.get("memory_id", ""),
                            content=result.get("processed_data", ""),
                            summary=result.get("summary", ""),
                            category=result.get("category_primary"),
                            composite_score=composite,
                            text_relevance_score=text_score,
                            entity_overlap_score=0.0,
                            graph_strength_score=0.0,
                            importance_score=result.get("importance_score", 0.5),
                            recency_score=0.5,
                            hop_distance=0,
                            graph_paths=[],
                            shared_entities=[],
                            timestamp=created_at,
                            access_count=result.get("access_count") or 0,
                            last_accessed=result.get("last_accessed"),
                        )
                    )

            logger.debug(
                f"TEXT_ONLY search returned {len(graph_results)} results for query: '{query_text[:50]}...'"
            )
            return graph_results

        except Exception as e:
            logger.error(f"TEXT_ONLY search failed: {e}")
            return []

    def _find_seed_memories(
        self,
        query_text: str,
        entities: List[str],
        categories: List[str],
        namespace: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Find initial seed memories for graph expansion"""

        seed_map: Dict[str, Dict[str, Any]] = {}

        def upsert_seed(memory_id: str, **metadata: Any) -> None:
            existing = seed_map.get(memory_id)
            if existing:
                # Keep the strongest signals from any strategy
                existing["text_score"] = max(
                    existing.get("text_score", 0.0), metadata.get("text_score", 0.0)
                )
                existing["entity_overlap_score"] = max(
                    existing.get("entity_overlap_score", 0.0),
                    metadata.get("entity_overlap_score", 0.0),
                )
                existing.setdefault("sources", set()).update(metadata.get("sources", set()))
                if metadata.get("matched_entities"):
                    existing["matched_entities"] = metadata["matched_entities"]
                if metadata.get("category"):
                    existing["category"] = metadata["category"]
                if metadata.get("created_at"):
                    existing["created_at"] = metadata["created_at"]
                if metadata.get("importance_score") is not None:
                    existing["importance_score"] = metadata["importance_score"]
            else:
                seed_map[memory_id] = {
                    "memory_id": memory_id,
                    "hop": 0,
                    "text_score": metadata.get("text_score", 0.0),
                    "entity_overlap_score": metadata.get("entity_overlap_score", 0.0),
                    "matched_entities": metadata.get("matched_entities", []),
                    "sources": metadata.get("sources", set()),
                    "created_at": metadata.get("created_at"),
                    "importance_score": metadata.get("importance_score"),
                    "category": metadata.get("category"),
                }

        # Strategy 1: entity-driven seeds
        if entities:
            query, params = self.query_builder.build_entity_search_query(
                entities=entities,
                namespace=namespace,
                min_relevance=0.3,
                limit=max(limit, len(entities) * 5),
            )

            with self.db_manager.get_session() as session:
                entity_results = self._execute_query(session, query, params)

            for row in entity_results:
                matched_entities = row.get("matched_entities")
                if isinstance(matched_entities, str):
                    matched_entities = [value.strip() for value in matched_entities.split(",") if value.strip()]
                entity_score = 0.0
                if entities:
                    entity_score = min(
                        1.0,
                        row.get("entity_match_count", 0) / max(1, len(entities)),
                    )

                upsert_seed(
                    row["memory_id"],
                    entity_overlap_score=entity_score,
                    matched_entities=matched_entities or [],
                    sources={"entity"},
                )

        # Strategy 2: text/category seeds via FTS/LIKE (SearchService)
        if query_text:
            text_seeds = self._find_text_based_seeds(
                query_text=query_text,
                namespace=namespace,
                categories=categories,
                limit=limit,
            )
            for seed in text_seeds:
                upsert_seed(
                    seed["memory_id"],
                    text_score=seed.get("text_score", 0.0),
                    importance_score=seed.get("importance_score"),
                    created_at=seed.get("created_at"),
                    category=seed.get("category"),
                    sources={"text"},
                )

        # Convert to list sorted by combined strength
        seeds: List[Dict[str, Any]] = []
        for memory_id, data in seed_map.items():
            sources = data.pop("sources", set())
            combined_score = max(data.get("text_score", 0.0), data.get("entity_overlap_score", 0.0))
            seeds.append(
                {
                    "memory_id": memory_id,
                    "hop": 0,
                    "cumulative_strength": max(0.3, combined_score),
                    "text_score": data.get("text_score", 0.0),
                    "entity_overlap_score": data.get("entity_overlap_score", 0.0),
                    "matched_entities": data.get("matched_entities", []),
                    "created_at": data.get("created_at"),
                    "importance_score": data.get("importance_score"),
                    "category": data.get("category"),
                    "sources": list(sources),
                }
            )

        seeds.sort(key=lambda s: (s.get("text_score", 0.0) + s.get("entity_overlap_score", 0.0)), reverse=True)
        return seeds[:limit]

    def _find_text_based_seeds(
        self,
        query_text: str,
        namespace: str,
        categories: Optional[List[str]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Find seed memories using text-based search when entity search fails

        This is a fallback for when:
        - No entities were extracted from the query
        - Entity search found no matching memories
        """
        if not query_text:
            return []

        try:
            # Use the search service for text-based search
            from .search_service import SearchService

            # Get a session from db_manager
            with self.db_manager.SessionLocal() as session:
                search_service = SearchService(
                    session=session,
                    database_type=self.dialect
                )

                results = search_service.search_memories(
                    query=query_text,
                    namespace=namespace,
                    limit=limit,
                    category_filter=categories,
                )

            # Convert to seed format
            seeds = []
            for result in results:
                if isinstance(result, dict) and "memory_id" in result:
                    seeds.append(
                        {
                            "memory_id": result["memory_id"],
                            "hop": 0,
                            "source": "text",
                            "text_score": result.get("composite_score")
                            or result.get("search_score", 0.0),
                            "importance_score": result.get("importance_score"),
                            "created_at": result.get("created_at"),
                            "category": result.get("category_primary"),
                        }
                    )

            logger.debug(
                f"Text-based seed search found {len(seeds)} memories for query: '{query_text[:50]}...'"
            )
            return seeds

        except Exception as e:
            logger.warning(f"Text-based seed search failed: {e}")
            return []

    def _expand_via_graph(
        self,
        seed_memory_ids: List[str],
        hop_distance: int,
        min_strength: float,
        namespace: str,
        limit_per_hop: int,
    ) -> List[Dict[str, Any]]:
        """Expand from seed memories via graph relationships"""

        if hop_distance == 0 or not seed_memory_ids:
            return []

        supports_recursive = getattr(self.query_builder, "supports_recursive_cte", lambda: False)()

        if supports_recursive:
            query, params = self.query_builder.build_graph_expansion_query(
                seed_memory_ids=seed_memory_ids,
                hop_distance=hop_distance,
                min_strength=min_strength,
                relationship_types=None,
                namespace=namespace,
                limit_per_hop=limit_per_hop,
            )

            with self.db_manager.get_session() as session:
                rows = self._execute_query(session, query, params)

            normalized: List[Dict[str, Any]] = []
            for row in rows:
                normalized.append(
                    {
                        "memory_id": row.get("memory_id"),
                        "hop": row.get("hop", 1),
                        "cumulative_strength": row.get("max_strength")
                        or row.get("cumulative_strength")
                        or row.get("edge_strength", 0.0),
                        "relationship_types": row.get("relationship_types", []),
                        "paths": row.get("paths"),
                    }
                )

            logger.debug(f"Graph expansion (recursive) found {len(normalized)} memories")
            return normalized

        # Iterative expansion for databases without recursive CTE support (SQLite/MySQL)
        visited = set(seed_memory_ids)
        frontier = list(seed_memory_ids)
        expansions: List[Dict[str, Any]] = []

        for hop in range(1, hop_distance + 1):
            if not frontier:
                break

            query, params = self.query_builder.build_graph_expansion_query(
                seed_memory_ids=frontier,
                hop_distance=1,
                min_strength=min_strength,
                relationship_types=None,
                namespace=namespace,
                limit_per_hop=limit_per_hop,
            )

            with self.db_manager.get_session() as session:
                rows = self._execute_query(session, query, params)

            next_frontier: List[str] = []
            hop_results: Dict[str, Dict[str, Any]] = {}

            for row in rows:
                candidate_id = row.get("memory_id")
                if not candidate_id or candidate_id in visited:
                    continue

                edge_strength = row.get("edge_strength") or row.get("cumulative_strength") or 0.0
                if edge_strength < min_strength:
                    continue

                adjusted_strength = min(1.0, edge_strength * (0.85 ** (hop - 1)))

                if candidate_id not in hop_results or adjusted_strength > hop_results[candidate_id]["cumulative_strength"]:
                    hop_results[candidate_id] = {
                        "memory_id": candidate_id,
                        "hop": hop,
                        "cumulative_strength": adjusted_strength,
                        "edge_strength": edge_strength,
                        "via": row.get("via_memory_id")
                        or row.get("source_memory_id")
                        or row.get("related_memory_id"),
                        "relationship_type": row.get("relationship_type"),
                    }

            # Order by strength and enforce per-hop limit
            ordered = sorted(
                hop_results.values(),
                key=lambda item: item["cumulative_strength"],
                reverse=True,
            )[:limit_per_hop]

            for item in ordered:
                expansions.append(item)
                visited.add(item["memory_id"])
                next_frontier.append(item["memory_id"])

            frontier = next_frontier

        logger.debug(f"Graph expansion (iterative) accumulated {len(expansions)} memories")
        return expansions

    def _enrich_with_memory_data(
        self,
        memory_refs: List[Dict[str, Any]],
        namespace: str,
    ) -> List[GraphSearchResult]:
        """Fetch full memory data and create GraphSearchResult objects"""

        if not memory_refs:
            return []

        memory_ids = [m["memory_id"] for m in memory_refs]

        with self.db_manager.get_session() as session:
            query = """
                SELECT memory_id, searchable_content as content, summary,
                       category_primary as category, importance_score,
                       created_at, access_count, last_accessed,
                       'short_term' as memory_type
                FROM short_term_memory
                WHERE memory_id IN ({})
                  AND namespace = ?

                UNION ALL

                SELECT memory_id, searchable_content as content, summary,
                       category_primary as category, importance_score,
                       created_at, access_count, last_accessed,
                       'long_term' as memory_type
                FROM long_term_memory
                WHERE memory_id IN ({})
                  AND namespace = ?
            """.format(
                ",".join(["?" for _ in memory_ids]),
                ",".join(["?" for _ in memory_ids]),
            )

            params = memory_ids + [namespace] + memory_ids + [namespace]
            memory_data = self._execute_query(session, query, {"__sqlite_params": params})

        memory_lookup = {m["memory_id"]: m for m in memory_data}

        def _calculate_recency_score(created_at: Any) -> float:
            try:
                if not created_at:
                    return 0.0
                if isinstance(created_at, str):
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    created_dt = created_at
                days_old = max(0, (datetime.now(tz=getattr(created_dt, "tzinfo", None)) - created_dt).days)
                return max(0.0, 1 - (days_old / 45))
            except Exception:
                return 0.0

        results: List[GraphSearchResult] = []
        for ref in memory_refs:
            mem = memory_lookup.get(ref["memory_id"])
            if not mem:
                continue

            category_value = mem.get("category")
            if category_value:
                from ..utils.pydantic_models import MemoryCategoryType

                try:
                    category_value = MemoryCategoryType(category_value)
                except (ValueError, AttributeError):
                    category_value = None

            created_at = mem.get("created_at")
            recency_score = _calculate_recency_score(created_at)

            text_score = ref.get("text_score", 0.0)
            entity_score = ref.get("entity_overlap_score", 0.0)
            graph_strength = ref.get("cumulative_strength", 0.0)

            combined_score = max(text_score, graph_strength, entity_score)

            result = GraphSearchResult(
                memory_id=ref["memory_id"],
                content=mem.get("content", ""),
                summary=mem.get("summary", ""),
                category=category_value,
                composite_score=max(combined_score, ref.get("composite_score", 0.0)),
                text_relevance_score=text_score,
                entity_overlap_score=entity_score,
                graph_strength_score=graph_strength,
                importance_score=mem.get("importance_score", 0.5),
                recency_score=recency_score,
                hop_distance=ref.get("hop", 0),
                graph_paths=[],
                shared_entities=ref.get("matched_entities", []),
                connected_via=[ref.get("via")] if ref.get("via") else [],
                match_reason=ref.get("match_reason", ""),
                timestamp=created_at,
                access_count=mem.get("access_count") or 0,
                last_accessed=mem.get("last_accessed"),
            )
            results.append(result)

        return results

    def _apply_composite_scoring(
        self,
        results: List[GraphSearchResult],
        weights: ScoringWeights,
    ) -> List[GraphSearchResult]:
        """Apply composite scoring with configurable weights"""

        for result in results:
            result.composite_score = (
                result.text_relevance_score * weights.text_relevance
                + result.entity_overlap_score * weights.entity_overlap
                + result.graph_strength_score * weights.graph_strength
                + result.importance_score * weights.importance
                + result.recency_score * weights.recency
            )

        return results

    def _generate_match_reasons(
        self,
        results: List[GraphSearchResult],
        query_text: str,
        entities: List[str],
    ) -> List[GraphSearchResult]:
        """Generate human-readable match explanations"""

        for result in results:
            reasons = []

            if result.hop_distance == 0:
                reasons.append("Direct match")
            else:
                reasons.append(f"{result.hop_distance}-hop connection")

            if result.shared_entities:
                reasons.append(
                    f"Shares entities: {', '.join(result.shared_entities[:3])}"
                )

            if result.entity_overlap_score > 0.7:
                reasons.append("Strong entity overlap")

            if result.text_relevance_score > 0.3:
                reasons.append("Textually relevant")

            if result.connected_via:
                reasons.append(f"Connected via {', '.join(result.connected_via)}")

            result.match_reason = " | ".join(reasons) if reasons else "Related memory"

        return results

    def _execute_query(
        self, session, query: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute database query and return results as dicts"""
        from sqlalchemy import text

        # Handle different parameter formats
        if "__mysql_params" in params or "__sqlite_params" in params:
            # Convert positional params (?) to named params for SQLAlchemy 2.0+
            param_list = params.get("__mysql_params") or params.get("__sqlite_params")

            # Convert ? placeholders to named parameters
            named_params = {}
            modified_query = query
            param_index = 0

            # Replace each ? with a named parameter
            while "?" in modified_query and param_index < len(param_list):
                param_name = f"param_{param_index}"
                modified_query = modified_query.replace("?", f":{param_name}", 1)
                named_params[param_name] = param_list[param_index]
                param_index += 1

            query_text = text(modified_query)
            result = session.execute(query_text, named_params)
        elif "__mongodb_pipeline" in params:
            # MongoDB aggregation - different handling
            # This would use pymongo instead
            raise NotImplementedError("MongoDB queries not yet implemented")
        else:
            # Named parameters (PostgreSQL style) - use text()
            query_text = text(query)
            result = session.execute(query_text, params)

        # Convert to list of dicts
        columns = result.keys() if hasattr(result, "keys") else []
        return [dict(zip(columns, row)) for row in result]

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return self.stats.copy()
