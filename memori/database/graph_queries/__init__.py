"""
Graph Query Builders for Database-Agnostic Graph Traversal
Provides query builders for PostgreSQL, MySQL, SQLite, and MongoDB
"""

from .base import GraphQueryBuilder
from .postgresql import PostgreSQLGraphQueryBuilder
from .mysql import MySQLGraphQueryBuilder
from .sqlite import SQLiteGraphQueryBuilder
from .mongodb import MongoDBGraphQueryBuilder


def get_query_builder(dialect: str) -> GraphQueryBuilder:
    """
    Factory function to get appropriate query builder for database dialect

    Args:
        dialect: Database dialect name (postgresql, mysql, sqlite, mongodb)

    Returns:
        Appropriate GraphQueryBuilder instance

    Raises:
        ValueError: If dialect is not supported
    """
    dialect = dialect.lower()

    if dialect == "postgresql":
        return PostgreSQLGraphQueryBuilder()
    elif dialect == "mysql":
        return MySQLGraphQueryBuilder()
    elif dialect == "sqlite":
        return SQLiteGraphQueryBuilder()
    elif dialect == "mongodb":
        return MongoDBGraphQueryBuilder()
    else:
        raise ValueError(
            f"Unsupported database dialect: {dialect}. "
            f"Supported: postgresql, mysql, sqlite, mongodb"
        )


__all__ = [
    "GraphQueryBuilder",
    "PostgreSQLGraphQueryBuilder",
    "MySQLGraphQueryBuilder",
    "SQLiteGraphQueryBuilder",
    "MongoDBGraphQueryBuilder",
    "get_query_builder",
]
