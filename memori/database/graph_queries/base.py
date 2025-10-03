"""
Base Graph Query Builder Interface
Defines the contract for all database-specific query builders
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class GraphQueryBuilder(ABC):
    """Abstract base class for database-specific graph query builders"""

    @abstractmethod
    def build_entity_search_query(
        self,
        entities: List[str],
        entity_types: Optional[List[str]] = None,
        namespace: str = "default",
        min_relevance: float = 0.0,
        limit: int = 50,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build query to find memories by entity values

        Args:
            entities: List of entity values to search for
            entity_types: Optional filter by entity types
            namespace: Memory namespace
            min_relevance: Minimum relevance score threshold
            limit: Maximum results

        Returns:
            Tuple of (query_string, params_dict)
        """
        pass

    @abstractmethod
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
        Build query to expand from seed memories via graph relationships

        Args:
            seed_memory_ids: Starting memory IDs
            hop_distance: Number of hops to traverse (1-3)
            min_strength: Minimum relationship strength
            relationship_types: Optional filter by relationship types
            namespace: Memory namespace
            limit_per_hop: Maximum results per hop level

        Returns:
            Tuple of (query_string, params_dict)
        """
        pass

    @abstractmethod
    def build_entity_cluster_query(
        self,
        entities: List[str],
        namespace: str = "default",
        min_shared_entities: int = 2,
        limit: int = 50,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build query to find memories that share multiple entities

        Args:
            entities: List of entity values
            namespace: Memory namespace
            min_shared_entities: Minimum number of shared entities
            limit: Maximum results

        Returns:
            Tuple of (query_string, params_dict)
        """
        pass

    @abstractmethod
    def build_relationship_discovery_query(
        self,
        memory_id: str,
        relationship_types: Optional[List[str]] = None,
        min_strength: float = 0.5,
        namespace: str = "default",
        limit: int = 20,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build query to find all direct relationships for a memory

        Args:
            memory_id: Memory ID to find relationships for
            relationship_types: Optional filter by types
            min_strength: Minimum relationship strength
            namespace: Memory namespace
            limit: Maximum results

        Returns:
            Tuple of (query_string, params_dict)
        """
        pass

    @abstractmethod
    def build_path_finding_query(
        self,
        source_memory_id: str,
        target_memory_id: str,
        max_depth: int = 3,
        namespace: str = "default",
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build query to find paths between two memories

        Args:
            source_memory_id: Starting memory
            target_memory_id: Destination memory
            max_depth: Maximum path length
            namespace: Memory namespace

        Returns:
            Tuple of (query_string, params_dict)
        """
        pass

    @abstractmethod
    def build_shared_entities_query(
        self,
        memory_id: str,
        namespace: str = "default",
        min_overlap: int = 1,
        limit: int = 50,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Build query to find memories sharing entities with given memory

        Args:
            memory_id: Memory ID to compare against
            namespace: Memory namespace
            min_overlap: Minimum number of shared entities
            limit: Maximum results

        Returns:
            Tuple of (query_string, params_dict)
        """
        pass

    def supports_recursive_cte(self) -> bool:
        """Whether this database supports recursive CTEs"""
        return True

    def get_parameter_placeholder(self, param_name: str) -> str:
        """
        Get database-specific parameter placeholder

        Args:
            param_name: Parameter name

        Returns:
            Placeholder string (e.g., ?, %s, :param_name)
        """
        return f":{param_name}"
