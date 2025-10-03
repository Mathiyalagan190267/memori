"""
MongoDB Graph Query Builder
Uses aggregation pipelines and $graphLookup for graph traversal
"""

from typing import Any, Dict, List, Optional

from .base import GraphQueryBuilder


class MongoDBGraphQueryBuilder(GraphQueryBuilder):
    """MongoDB-specific graph query builder using aggregation framework"""

    def build_entity_search_query(
        self,
        entities: List[str],
        entity_types: Optional[List[str]] = None,
        namespace: str = "default",
        min_relevance: float = 0.0,
        limit: int = 50,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build MongoDB aggregation pipeline for entity search"""

        # Normalize entities
        normalized_entities = [e.lower() for e in entities]

        # Build aggregation pipeline
        pipeline = [
            {
                "$match": {
                    "namespace": namespace,
                    "normalized_value": {"$in": normalized_entities},
                    "relevance_score": {"$gte": min_relevance},
                }
            }
        ]

        if entity_types:
            pipeline[0]["$match"]["entity_type"] = {"$in": entity_types}

        pipeline.extend(
            [
                {
                    "$group": {
                        "_id": {"memory_id": "$memory_id", "memory_type": "$memory_type"},
                        "entity_match_count": {"$sum": 1},
                        "avg_relevance": {"$avg": "$relevance_score"},
                        "matched_entities": {"$addToSet": "$entity_value"},
                    }
                },
                {"$sort": {"entity_match_count": -1, "avg_relevance": -1}},
                {"$limit": limit},
                {
                    "$project": {
                        "memory_id": "$_id.memory_id",
                        "memory_type": "$_id.memory_type",
                        "entity_match_count": 1,
                        "avg_relevance": 1,
                        "matched_entities": 1,
                        "_id": 0,
                    }
                },
            ]
        )

        # MongoDB doesn't use traditional params, return pipeline
        return {"__mongodb_pipeline": pipeline}, {}

    def build_graph_expansion_query(
        self,
        seed_memory_ids: List[str],
        hop_distance: int,
        min_strength: float,
        relationship_types: Optional[List[str]] = None,
        namespace: str = "default",
        limit_per_hop: int = 10,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build MongoDB $graphLookup pipeline for graph expansion"""

        match_condition = {
            "namespace": namespace,
            "strength": {"$gte": min_strength},
        }

        if relationship_types:
            match_condition["relationship_type"] = {"$in": relationship_types}

        pipeline = [
            {"$match": {"memory_id": {"$in": seed_memory_ids}}},
            {
                "$graphLookup": {
                    "from": "memory_relationships",
                    "startWith": "$memory_id",
                    "connectFromField": "target_memory_id",
                    "connectToField": "source_memory_id",
                    "as": "graph_path",
                    "maxDepth": hop_distance - 1,
                    "depthField": "hop",
                    "restrictSearchWithMatch": match_condition,
                }
            },
            {"$unwind": "$graph_path"},
            {
                "$group": {
                    "_id": "$graph_path.target_memory_id",
                    "hop": {"$min": "$graph_path.hop"},
                    "cumulative_strength": {"$max": "$graph_path.strength"},
                    "relationship_types": {"$addToSet": "$graph_path.relationship_type"},
                }
            },
            {"$sort": {"hop": 1, "cumulative_strength": -1}},
            {"$limit": limit_per_hop * hop_distance},
            {
                "$project": {
                    "memory_id": "$_id",
                    "hop": {"$add": ["$hop", 1]},  # Adjust hop count
                    "cumulative_strength": 1,
                    "relationship_types": 1,
                    "_id": 0,
                }
            },
        ]

        return {"__mongodb_pipeline": pipeline}, {}

    def build_entity_cluster_query(
        self,
        entities: List[str],
        namespace: str = "default",
        min_shared_entities: int = 2,
        limit: int = 50,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Find memories sharing multiple entities"""

        normalized_entities = [e.lower() for e in entities]

        pipeline = [
            {
                "$match": {
                    "namespace": namespace,
                    "normalized_value": {"$in": normalized_entities},
                }
            },
            {
                "$group": {
                    "_id": {"memory_id": "$memory_id", "memory_type": "$memory_type"},
                    "shared_entity_count": {"$addToSet": "$normalized_value"},
                    "avg_relevance": {"$avg": "$relevance_score"},
                    "shared_entities": {"$addToSet": "$entity_value"},
                }
            },
            {"$match": {"shared_entity_count": {"$gte": min_shared_entities}}},
            {
                "$addFields": {
                    "shared_entity_count": {"$size": "$shared_entity_count"}
                }
            },
            {"$sort": {"shared_entity_count": -1, "avg_relevance": -1}},
            {"$limit": limit},
            {
                "$project": {
                    "memory_id": "$_id.memory_id",
                    "memory_type": "$_id.memory_type",
                    "shared_entity_count": 1,
                    "avg_relevance": 1,
                    "shared_entities": 1,
                    "_id": 0,
                }
            },
        ]

        return {"__mongodb_pipeline": pipeline}, {}

    def build_relationship_discovery_query(
        self,
        memory_id: str,
        relationship_types: Optional[List[str]] = None,
        min_strength: float = 0.5,
        namespace: str = "default",
        limit: int = 20,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Find all direct relationships for a memory"""

        match_condition = {
            "$or": [
                {"source_memory_id": memory_id},
                {"target_memory_id": memory_id},
            ],
            "namespace": namespace,
            "strength": {"$gte": min_strength},
        }

        if relationship_types:
            match_condition["relationship_type"] = {"$in": relationship_types}

        pipeline = [
            {"$match": match_condition},
            {"$sort": {"strength": -1}},
            {"$limit": limit},
            {
                "$addFields": {
                    "related_memory_id": {
                        "$cond": {
                            "if": {"$eq": ["$source_memory_id", memory_id]},
                            "then": "$target_memory_id",
                            "else": "$source_memory_id",
                        }
                    }
                }
            },
        ]

        return {"__mongodb_pipeline": pipeline}, {}

    def build_path_finding_query(
        self,
        source_memory_id: str,
        target_memory_id: str,
        max_depth: int = 3,
        namespace: str = "default",
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Find paths between two memories using $graphLookup"""

        pipeline = [
            {"$match": {"source_memory_id": source_memory_id, "namespace": namespace}},
            {
                "$graphLookup": {
                    "from": "memory_relationships",
                    "startWith": "$target_memory_id",
                    "connectFromField": "target_memory_id",
                    "connectToField": "source_memory_id",
                    "as": "path",
                    "maxDepth": max_depth - 1,
                    "depthField": "depth",
                }
            },
            {
                "$match": {
                    "path.target_memory_id": target_memory_id,
                }
            },
            {"$limit": 5},
            {
                "$project": {
                    "path": 1,
                    "depth": {"$size": "$path"},
                    "total_strength": {"$multiply": ["$strength", "$path.strength"]},
                }
            },
            {"$sort": {"depth": 1, "total_strength": -1}},
        ]

        return {"__mongodb_pipeline": pipeline}, {}

    def build_shared_entities_query(
        self,
        memory_id: str,
        namespace: str = "default",
        min_overlap: int = 1,
        limit: int = 50,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Find memories sharing entities with given memory"""

        pipeline = [
            # Step 1: Get source memory entities
            {
                "$match": {
                    "memory_id": memory_id,
                    "namespace": namespace,
                }
            },
            {
                "$group": {
                    "_id": None,
                    "source_entities": {"$addToSet": "$normalized_value"},
                }
            },
            # Step 2: Find memories with matching entities
            {
                "$lookup": {
                    "from": "memory_entities",
                    "let": {"source_ents": "$source_entities"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$and": [
                                        {"$ne": ["$memory_id", memory_id]},
                                        {"$eq": ["$namespace", namespace]},
                                        {"$in": ["$normalized_value", "$$source_ents"]},
                                    ]
                                }
                            }
                        },
                        {
                            "$group": {
                                "_id": {
                                    "memory_id": "$memory_id",
                                    "memory_type": "$memory_type",
                                },
                                "shared_count": {"$sum": 1},
                                "avg_relevance": {"$avg": "$relevance_score"},
                                "shared_entities": {"$addToSet": "$entity_value"},
                            }
                        },
                        {"$match": {"shared_count": {"$gte": min_overlap}}},
                        {"$sort": {"shared_count": -1, "avg_relevance": -1}},
                        {"$limit": limit},
                    ],
                    "as": "matches",
                }
            },
            {"$unwind": "$matches"},
            {"$replaceRoot": {"newRoot": "$matches"}},
            {
                "$project": {
                    "memory_id": "$_id.memory_id",
                    "memory_type": "$_id.memory_type",
                    "shared_count": 1,
                    "avg_relevance": 1,
                    "shared_entities": 1,
                    "_id": 0,
                }
            },
        ]

        return {"__mongodb_pipeline": pipeline}, {}

    def get_parameter_placeholder(self, param_name: str) -> str:
        """MongoDB doesn't use SQL-style placeholders"""
        return f"${param_name}"

    def supports_recursive_cte(self) -> bool:
        """MongoDB doesn't use CTEs, but has $graphLookup"""
        return False
