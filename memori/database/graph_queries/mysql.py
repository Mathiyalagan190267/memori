"""
MySQL Graph Query Builder
Uses temporary tables and iterative queries for graph traversal
"""

from typing import Any, Dict, List, Optional

from .base import GraphQueryBuilder


class MySQLGraphQueryBuilder(GraphQueryBuilder):
    """MySQL-specific graph query builder"""

    def supports_recursive_cte(self) -> bool:  # type: ignore[override]
        return False

    def build_entity_search_query(
        self,
        entities: List[str],
        entity_types: Optional[List[str]] = None,
        namespace: str = "default",
        min_relevance: float = 0.0,
        limit: int = 50,
    ) -> tuple[str, Dict[str, Any]]:
        """Find memories by entity values"""

        normalized_entities = [e.lower() for e in entities]

        # Build IN clause for entities
        entity_placeholders = ",".join([f"%s" for _ in normalized_entities])

        query = f"""
        SELECT
            me.memory_id,
            me.memory_type,
            COUNT(*) as entity_match_count,
            AVG(me.relevance_score) as avg_relevance,
            GROUP_CONCAT(DISTINCT me.entity_value) as matched_entities
        FROM memory_entities me
        WHERE me.namespace = %s
            AND me.normalized_value IN ({entity_placeholders})
            AND me.relevance_score >= %s
        """

        params_list = [namespace] + normalized_entities + [min_relevance]

        if entity_types:
            type_placeholders = ",".join(["%s" for _ in entity_types])
            query += f" AND me.entity_type IN ({type_placeholders})"
            params_list.extend(entity_types)

        query += """
        GROUP BY me.memory_id, me.memory_type
        ORDER BY entity_match_count DESC, avg_relevance DESC
        LIMIT %s
        """

        params_list.append(limit)

        # Convert to dict for consistency
        params = {"__mysql_params": params_list}

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
        """
        Note: MySQL recursive CTE support is limited
        This returns a query for 1-hop expansion
        Multi-hop should be handled iteratively by the service layer
        """

        seed_placeholders = ",".join(["%s" for _ in seed_memory_ids])

        query = f"""
        SELECT DISTINCT
            CASE
                WHEN mr.source_memory_id IN ({seed_placeholders}) THEN mr.target_memory_id
                ELSE mr.source_memory_id
            END as memory_id,
            CASE
                WHEN mr.source_memory_id IN ({seed_placeholders}) THEN mr.source_memory_id
                ELSE mr.target_memory_id
            END as via_memory_id,
            mr.strength as edge_strength,
            mr.relationship_type
        FROM memory_relationships mr
        WHERE (mr.source_memory_id IN ({seed_placeholders})
            OR mr.target_memory_id IN ({seed_placeholders}))
            AND mr.namespace = %s
            AND mr.strength >= %s
        """

        params_list = (
            seed_memory_ids
            + seed_memory_ids
            + seed_memory_ids
            + seed_memory_ids
            + [namespace, min_strength]
        )

        if relationship_types:
            type_placeholders = ",".join(["%s" for _ in relationship_types])
            query += f" AND mr.relationship_type IN ({type_placeholders})"
            params_list.extend(relationship_types)

        query += """
        ORDER BY edge_strength DESC
        LIMIT %s
        """

        params_list.append(limit_per_hop)

        params = {"__mysql_params": params_list}

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
        entity_placeholders = ",".join(["%s" for _ in normalized_entities])

        query = f"""
        SELECT
            me.memory_id,
            me.memory_type,
            COUNT(DISTINCT me.normalized_value) as shared_entity_count,
            AVG(me.relevance_score) as avg_relevance,
            GROUP_CONCAT(DISTINCT me.entity_value) as shared_entities
        FROM memory_entities me
        WHERE me.namespace = %s
            AND me.normalized_value IN ({entity_placeholders})
        GROUP BY me.memory_id, me.memory_type
        HAVING COUNT(DISTINCT me.normalized_value) >= %s
        ORDER BY shared_entity_count DESC, avg_relevance DESC
        LIMIT %s
        """

        params_list = [namespace] + normalized_entities + [min_shared_entities, limit]
        params = {"__mysql_params": params_list}

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
                WHEN mr.source_memory_id = %s THEN mr.target_memory_id
                ELSE mr.source_memory_id
            END as related_memory_id
        FROM memory_relationships mr
        WHERE (mr.source_memory_id = %s OR mr.target_memory_id = %s)
            AND mr.namespace = %s
            AND mr.strength >= %s
        """

        params_list = [memory_id, memory_id, memory_id, namespace, min_strength]

        if relationship_types:
            type_placeholders = ",".join(["%s" for _ in relationship_types])
            query += f" AND mr.relationship_type IN ({type_placeholders})"
            params_list.extend(relationship_types)

        query += """
        ORDER BY mr.strength DESC
        LIMIT %s
        """

        params_list.append(limit)
        params = {"__mysql_params": params_list}

        return query, params

    def build_path_finding_query(
        self,
        source_memory_id: str,
        target_memory_id: str,
        max_depth: int = 3,
        namespace: str = "default",
    ) -> tuple[str, Dict[str, Any]]:
        """Note: Path finding in MySQL requires iterative queries"""

        # Return simple 1-hop check
        query = """
        SELECT
            source_memory_id,
            target_memory_id,
            relationship_type,
            strength,
            1 as depth
        FROM memory_relationships
        WHERE ((source_memory_id = %s AND target_memory_id = %s)
            OR (source_memory_id = %s AND target_memory_id = %s))
            AND namespace = %s
        LIMIT 1
        """

        params_list = [source_memory_id, target_memory_id, target_memory_id, source_memory_id, namespace]
        params = {"__mysql_params": params_list}

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
        SELECT
            me2.memory_id,
            me2.memory_type,
            COUNT(DISTINCT me2.normalized_value) as shared_count,
            AVG(me2.relevance_score) as avg_relevance,
            GROUP_CONCAT(DISTINCT me2.entity_value) as shared_entities
        FROM memory_entities me1
        JOIN memory_entities me2 ON me1.normalized_value = me2.normalized_value
        WHERE me1.memory_id = %s
            AND me2.memory_id != %s
            AND me1.namespace = %s
            AND me2.namespace = %s
        GROUP BY me2.memory_id, me2.memory_type
        HAVING COUNT(DISTINCT me2.normalized_value) >= %s
        ORDER BY shared_count DESC, avg_relevance DESC
        LIMIT %s
        """

        params_list = [memory_id, memory_id, namespace, namespace, min_overlap, limit]
        params = {"__mysql_params": params_list}

        return query, params

    def get_parameter_placeholder(self, param_name: str) -> str:
        """MySQL uses %s placeholders"""
        return "%s"

    def supports_recursive_cte(self) -> bool:
        """MySQL 8.0+ supports recursive CTEs but with limitations"""
        return False  # Return False to use iterative approach
