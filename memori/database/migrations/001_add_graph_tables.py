"""
Migration: Add Graph Tables for Entity and Relationship Storage
Version: 001
Date: 2025-10-03
Description: Adds memory_entities and memory_relationships tables for graph-based search
"""

from datetime import datetime
from loguru import logger


def upgrade(engine):
    """Add graph tables to existing database"""
    dialect = engine.dialect.name
    logger.info(f"Running migration 001 on {dialect} database")

    with engine.connect() as conn:
        try:
            # Step 1: Create memory_entities table
            if dialect == "sqlite":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_entities (
                        entity_id TEXT PRIMARY KEY,
                        memory_id TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_value TEXT NOT NULL,
                        normalized_value TEXT NOT NULL,
                        relevance_score REAL DEFAULT 0.5,
                        namespace TEXT NOT NULL DEFAULT 'default',
                        frequency INTEGER DEFAULT 1,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        context TEXT
                    )
                    """
                )
            elif dialect == "mysql":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_entities (
                        entity_id VARCHAR(255) PRIMARY KEY,
                        memory_id VARCHAR(255) NOT NULL,
                        memory_type VARCHAR(50) NOT NULL,
                        entity_type VARCHAR(100) NOT NULL,
                        entity_value VARCHAR(500) NOT NULL,
                        normalized_value VARCHAR(500) NOT NULL,
                        relevance_score FLOAT DEFAULT 0.5,
                        namespace VARCHAR(255) NOT NULL DEFAULT 'default',
                        frequency INT DEFAULT 1,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        context TEXT
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
            elif dialect == "postgresql":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_entities (
                        entity_id VARCHAR(255) PRIMARY KEY,
                        memory_id VARCHAR(255) NOT NULL,
                        memory_type VARCHAR(50) NOT NULL,
                        entity_type VARCHAR(100) NOT NULL,
                        entity_value VARCHAR(500) NOT NULL,
                        normalized_value VARCHAR(500) NOT NULL,
                        relevance_score FLOAT DEFAULT 0.5,
                        namespace VARCHAR(255) NOT NULL DEFAULT 'default',
                        frequency INTEGER DEFAULT 1,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        context TEXT
                    )
                    """
                )

            # Step 2: Create entity indexes
            _create_entity_indexes(conn, dialect)

            # Step 3: Create memory_relationships table
            if dialect == "sqlite":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_relationships (
                        relationship_id TEXT PRIMARY KEY,
                        source_memory_id TEXT NOT NULL,
                        target_memory_id TEXT NOT NULL,
                        source_memory_type TEXT NOT NULL,
                        target_memory_type TEXT NOT NULL,
                        relationship_type TEXT NOT NULL,
                        strength REAL NOT NULL DEFAULT 0.5,
                        bidirectional INTEGER DEFAULT 1,
                        namespace TEXT NOT NULL DEFAULT 'default',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_strengthened TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        reasoning TEXT,
                        shared_entity_count INTEGER DEFAULT 0,
                        metadata_json TEXT
                    )
                    """
                )
            elif dialect == "mysql":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_relationships (
                        relationship_id VARCHAR(255) PRIMARY KEY,
                        source_memory_id VARCHAR(255) NOT NULL,
                        target_memory_id VARCHAR(255) NOT NULL,
                        source_memory_type VARCHAR(50) NOT NULL,
                        target_memory_type VARCHAR(50) NOT NULL,
                        relationship_type VARCHAR(100) NOT NULL,
                        strength FLOAT NOT NULL DEFAULT 0.5,
                        bidirectional BOOLEAN DEFAULT TRUE,
                        namespace VARCHAR(255) NOT NULL DEFAULT 'default',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_strengthened TIMESTAMP NULL,
                        access_count INT DEFAULT 0,
                        reasoning TEXT,
                        shared_entity_count INT DEFAULT 0,
                        metadata_json JSON
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
            elif dialect == "postgresql":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_relationships (
                        relationship_id VARCHAR(255) PRIMARY KEY,
                        source_memory_id VARCHAR(255) NOT NULL,
                        target_memory_id VARCHAR(255) NOT NULL,
                        source_memory_type VARCHAR(50) NOT NULL,
                        target_memory_type VARCHAR(50) NOT NULL,
                        relationship_type VARCHAR(100) NOT NULL,
                        strength FLOAT NOT NULL DEFAULT 0.5,
                        bidirectional BOOLEAN DEFAULT TRUE,
                        namespace VARCHAR(255) NOT NULL DEFAULT 'default',
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_strengthened TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        reasoning TEXT,
                        shared_entity_count INTEGER DEFAULT 0,
                        metadata_json JSONB
                    )
                    """
                )

            # Step 4: Create relationship indexes
            _create_relationship_indexes(conn, dialect)

            conn.commit()
            logger.success("Migration 001 completed successfully")

        except Exception as e:
            logger.error(f"Migration 001 failed: {e}")
            conn.rollback()
            raise


def downgrade(engine):
    """Remove graph tables"""
    dialect = engine.dialect.name
    logger.info(f"Reverting migration 001 on {dialect} database")

    with engine.connect() as conn:
        try:
            conn.execute("DROP TABLE IF EXISTS memory_relationships")
            conn.execute("DROP TABLE IF EXISTS memory_entities")
            conn.commit()
            logger.success("Migration 001 reverted successfully")

        except Exception as e:
            logger.error(f"Migration 001 revert failed: {e}")
            conn.rollback()
            raise


def _create_entity_indexes(conn, dialect):
    """Create indexes for memory_entities table"""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_entity_memory ON memory_entities(memory_id, memory_type)",
        "CREATE INDEX IF NOT EXISTS idx_entity_type ON memory_entities(entity_type)",
        "CREATE INDEX IF NOT EXISTS idx_entity_value ON memory_entities(entity_value)",
        "CREATE INDEX IF NOT EXISTS idx_entity_normalized ON memory_entities(normalized_value)",
        "CREATE INDEX IF NOT EXISTS idx_entity_namespace ON memory_entities(namespace)",
        "CREATE INDEX IF NOT EXISTS idx_entity_relevance ON memory_entities(relevance_score)",
        "CREATE INDEX IF NOT EXISTS idx_entity_type_value ON memory_entities(entity_type, normalized_value)",
        "CREATE INDEX IF NOT EXISTS idx_entity_namespace_type ON memory_entities(namespace, entity_type)",
    ]

    # Compound index for optimal queries
    if dialect == "sqlite":
        indexes.append(
            "CREATE INDEX IF NOT EXISTS idx_entity_compound ON memory_entities(namespace, entity_type, normalized_value, relevance_score)"
        )
    else:
        indexes.append(
            "CREATE INDEX idx_entity_compound ON memory_entities(namespace, entity_type, normalized_value, relevance_score)"
        )

    for index_sql in indexes:
        try:
            conn.execute(index_sql)
        except Exception as e:
            logger.warning(f"Could not create index: {e}")


def _create_relationship_indexes(conn, dialect):
    """Create indexes for memory_relationships table"""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_rel_source ON memory_relationships(source_memory_id, source_memory_type)",
        "CREATE INDEX IF NOT EXISTS idx_rel_target ON memory_relationships(target_memory_id, target_memory_type)",
        "CREATE INDEX IF NOT EXISTS idx_rel_type ON memory_relationships(relationship_type)",
        "CREATE INDEX IF NOT EXISTS idx_rel_strength ON memory_relationships(strength)",
        "CREATE INDEX IF NOT EXISTS idx_rel_namespace ON memory_relationships(namespace)",
        "CREATE INDEX IF NOT EXISTS idx_rel_bidirectional ON memory_relationships(bidirectional)",
        "CREATE INDEX IF NOT EXISTS idx_rel_source_type ON memory_relationships(source_memory_id, relationship_type)",
        "CREATE INDEX IF NOT EXISTS idx_rel_target_type ON memory_relationships(target_memory_id, relationship_type)",
        "CREATE INDEX IF NOT EXISTS idx_rel_entity_count ON memory_relationships(shared_entity_count)",
    ]

    # Compound indexes for graph traversal
    if dialect == "sqlite":
        indexes.extend(
            [
                "CREATE INDEX IF NOT EXISTS idx_rel_compound_source ON memory_relationships(source_memory_id, relationship_type, strength)",
                "CREATE INDEX IF NOT EXISTS idx_rel_compound_target ON memory_relationships(target_memory_id, relationship_type, strength)",
                "CREATE INDEX IF NOT EXISTS idx_rel_namespace_type ON memory_relationships(namespace, relationship_type, strength)",
            ]
        )
    else:
        indexes.extend(
            [
                "CREATE INDEX idx_rel_compound_source ON memory_relationships(source_memory_id, relationship_type, strength)",
                "CREATE INDEX idx_rel_compound_target ON memory_relationships(target_memory_id, relationship_type, strength)",
                "CREATE INDEX idx_rel_namespace_type ON memory_relationships(namespace, relationship_type, strength)",
            ]
        )

    for index_sql in indexes:
        try:
            conn.execute(index_sql)
        except Exception as e:
            logger.warning(f"Could not create index: {e}")


def get_version():
    """Return migration version"""
    return "001"


def get_description():
    """Return migration description"""
    return "Add graph tables for entity and relationship storage"
