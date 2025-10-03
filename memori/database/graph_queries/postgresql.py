"""
PostgreSQL Graph Query Builder
Uses recursive CTEs for efficient graph traversal
"""

from typing import Any, Dict, List, Optional

from .base import GraphQueryBuilder


class PostgreSQLGraphQueryBuilder(GraphQueryBuilder):
    """PostgreSQL-specific graph query builder with recursive CTEs"""

    def build_entity_search_query(
        self,
        entities: List[str],
        entity_types: Optional[List[str]] = None,
        namespace: str = "default",
        min_relevance: float = 0.0,
        limit: int = 50,
    ) -> tuple[str, Dict[str, Any]]:
        """Find memories by entity values"""

        # Normalize entities for case-insensitive search
        normalized_entities = [e.lower() for e in entities]

        query = """
        SELECT DISTINCT
            me.memory_id,
            me.memory_type,
            COUNT(*) as entity_match_count,
            AVG(me.relevance_score) as avg_relevance,
            array_agg(DISTINCT me.entity_value) as matched_entities
        FROM memory_entities me
        WHERE me.namespace = :namespace
            AND me.normalized_value = ANY(:entities)
            AND me.relevance_score >= :min_relevance
        """

        if entity_types:
            query += " AND me.entity_type = ANY(:entity_types)"

        query += """
        GROUP BY me.memory_id, me.memory_type
        ORDER BY entity_match_count DESC, avg_relevance DESC
        LIMIT :limit
        """

        params = {
            "namespace": namespace,
            "entities": normalized_entities,
            "min_relevance": min_relevance,
            "limit": limit,
        }

        if entity_types:
            params["entity_types"] = entity_types

        return query, params

    def build_graph_expansion_query(
        self,
        seed_memory_ids: List[str],
        hop_distance: int,
        min_strength: float,
        relationship_types: Optional[List[str]] = None,
        namespace: str = "default",
        limit_per_hop: int = 10,
    ) -> tuple[str, Dict[str, Any]]:
        """Expand via graph relationships using recursive CTE"""

        if not seed_memory_ids:
            return "SELECT NULL WHERE FALSE", {}

        seed_placeholders = ", ".join(
            [f":seed_{i}" for i in range(len(seed_memory_ids))]
        )

        query = f"""
        WITH RECURSIVE graph_walk AS (
            -- Base case: seed memories at hop 0
            SELECT
                seed_id as memory_id,
                0 as hop,
                1.0::double precision as cumulative_strength,
                ARRAY[seed_id]::text[] as path,
                NULL::text as relationship_type
            FROM unnest(ARRAY[{seed_placeholders}]) as seed_id

            UNION ALL

            -- Recursive case: follow relationships
            SELECT
                CASE
                    WHEN mr.source_memory_id = gw.memory_id THEN mr.target_memory_id
                    ELSE mr.source_memory_id
                END as memory_id,
                gw.hop + 1 as hop,
                gw.cumulative_strength * mr.strength as cumulative_strength,
                gw.path || CASE
                    WHEN mr.source_memory_id = gw.memory_id THEN mr.target_memory_id
                    ELSE mr.source_memory_id
                END as path,
                mr.relationship_type
            FROM graph_walk gw
            JOIN memory_relationships mr ON (
                (mr.source_memory_id = gw.memory_id OR mr.target_memory_id = gw.memory_id)
                AND mr.namespace = :namespace
                AND mr.strength >= :min_strength
        """

        if relationship_types:
            query += " AND mr.relationship_type = ANY(:rel_types)"

        query += """
            )
            WHERE gw.hop < :max_hops
                AND NOT (
                    CASE
                        WHEN mr.source_memory_id = gw.memory_id THEN mr.target_memory_id
                        ELSE mr.source_memory_id
                    END = ANY(gw.path)
                )
        )
        SELECT DISTINCT
            memory_id,
            hop,
            MAX(cumulative_strength) as max_strength,
            array_agg(DISTINCT relationship_type) FILTER (WHERE relationship_type IS NOT NULL) as relationship_types,
            array_agg(DISTINCT path) as paths
        FROM graph_walk
        WHERE hop > 0
        GROUP BY memory_id, hop
        ORDER BY hop ASC, max_strength DESC
        LIMIT :limit
        """

        params = {
            "namespace": namespace,
            "min_strength": min_strength,
            "max_hops": hop_distance,
            "limit": limit_per_hop * hop_distance,
        }

        for i, seed in enumerate(seed_memory_ids):
            params[f"seed_{i}"] = seed

        if relationship_types:
            params["rel_types"] = relationship_types

        return query, params

    def build_entity_cluster_query(
        self,
        entities: List[str],
        namespace: str = "default",
        min_shared_entities: int = 2,
        limit: int = 50,
    ) -> tuple[str, Dict[str, Any]]:
        """Find memories sharing multiple entities"""

        normalized_entities = [e.lower() for e in entities]

        query = """
        SELECT
            me.memory_id,
            me.memory_type,
            COUNT(DISTINCT me.normalized_value) as shared_entity_count,
            AVG(me.relevance_score) as avg_relevance,
            array_agg(DISTINCT me.entity_value) as shared_entities
        FROM memory_entities me
        WHERE me.namespace = :namespace
            AND me.normalized_value = ANY(:entities)
        GROUP BY me.memory_id, me.memory_type
        HAVING COUNT(DISTINCT me.normalized_value) >= :min_shared
        ORDER BY shared_entity_count DESC, avg_relevance DESC
        LIMIT :limit
        """

        params = {
            "namespace": namespace,
            "entities": normalized_entities,
            "min_shared": min_shared_entities,
            "limit": limit,
        }

        return query, params

    def build_relationship_discovery_query(
        self,
        memory_id: str,
        relationship_types: Optional[List[str]] = None,
        min_strength: float = 0.5,
        namespace: str = "default",
        limit: int = 20,
    ) -> tuple[str, Dict[str, Any]]:
        """Find all direct relationships for a memory"""

        query = """
        SELECT
            mr.relationship_id,
            mr.source_memory_id,
            mr.target_memory_id,
            mr.relationship_type,
            mr.strength,
            mr.reasoning,
            mr.shared_entity_count,
            CASE
                WHEN mr.source_memory_id = :memory_id THEN mr.target_memory_id
                ELSE mr.source_memory_id
            END as related_memory_id
        FROM memory_relationships mr
        WHERE (mr.source_memory_id = :memory_id OR mr.target_memory_id = :memory_id)
            AND mr.namespace = :namespace
            AND mr.strength >= :min_strength
        """

        if relationship_types:
            query += " AND mr.relationship_type = ANY(:rel_types)"

        query += """
        ORDER BY mr.strength DESC
        LIMIT :limit
        """

        params = {
            "memory_id": memory_id,
            "namespace": namespace,
            "min_strength": min_strength,
            "limit": limit,
        }

        if relationship_types:
            params["rel_types"] = relationship_types

        return query, params

    def build_path_finding_query(
        self,
        source_memory_id: str,
        target_memory_id: str,
        max_depth: int = 3,
        namespace: str = "default",
    ) -> tuple[str, Dict[str, Any]]:
        """Find paths between two memories using recursive CTE"""

        query = """
        WITH RECURSIVE path_search AS (
            -- Base case: start from source
            SELECT
                :source as current_id,
                :target as target_id,
                ARRAY[:source]::text[] as path,
                ARRAY[]::text[] as relationship_path,
                0 as depth,
                1.0::double precision as total_strength

            UNION ALL

            -- Recursive case: follow relationships
            SELECT
                CASE
                    WHEN mr.source_memory_id = ps.current_id THEN mr.target_memory_id
                    ELSE mr.source_memory_id
                END as current_id,
                ps.target_id,
                ps.path || CASE
                    WHEN mr.source_memory_id = ps.current_id THEN mr.target_memory_id
                    ELSE mr.source_memory_id
                END as path,
                ps.relationship_path || mr.relationship_type as relationship_path,
                ps.depth + 1 as depth,
                ps.total_strength * mr.strength as total_strength
            FROM path_search ps
            JOIN memory_relationships mr ON (
                (mr.source_memory_id = ps.current_id OR mr.target_memory_id = ps.current_id)
                AND mr.namespace = :namespace
            )
            WHERE ps.depth < :max_depth
                AND ps.current_id != ps.target_id
                AND NOT (
                    CASE
                        WHEN mr.source_memory_id = ps.current_id THEN mr.target_memory_id
                        ELSE mr.source_memory_id
                    END = ANY(ps.path)
                )
        )
        SELECT
            path,
            relationship_path,
            depth,
            total_strength
        FROM path_search
        WHERE current_id = target_id
        ORDER BY depth ASC, total_strength DESC
        LIMIT 5
        """

        params = {
            "source": source_memory_id,
            "target": target_memory_id,
            "namespace": namespace,
            "max_depth": max_depth,
        }

        return query, params

    def build_shared_entities_query(
        self,
        memory_id: str,
        namespace: str = "default",
        min_overlap: int = 1,
        limit: int = 50,
    ) -> tuple[str, Dict[str, Any]]:
        """Find memories sharing entities with given memory"""

        query = """
        WITH source_entities AS (
            SELECT normalized_value
            FROM memory_entities
            WHERE memory_id = :memory_id
                AND namespace = :namespace
        )
        SELECT
            me.memory_id,
            me.memory_type,
            COUNT(DISTINCT me.normalized_value) as shared_count,
            AVG(me.relevance_score) as avg_relevance,
            array_agg(DISTINCT me.entity_value) as shared_entities
        FROM memory_entities me
        JOIN source_entities se ON me.normalized_value = se.normalized_value
        WHERE me.memory_id != :memory_id
            AND me.namespace = :namespace
        GROUP BY me.memory_id, me.memory_type
        HAVING COUNT(DISTINCT me.normalized_value) >= :min_overlap
        ORDER BY shared_count DESC, avg_relevance DESC
        LIMIT :limit
        """

        params = {
            "memory_id": memory_id,
            "namespace": namespace,
            "min_overlap": min_overlap,
            "limit": limit,
        }

        return query, params

    def get_parameter_placeholder(self, param_name: str) -> str:
        """PostgreSQL uses :param_name style placeholders"""
        return f":{param_name}"
