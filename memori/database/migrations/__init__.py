"""
Database Migration Manager for Memori
Handles schema migrations across different database backends
"""

import importlib
import os
from pathlib import Path
from typing import List

from loguru import logger


class MigrationManager:
    """Manages database schema migrations"""

    def __init__(self, engine):
        """
        Initialize migration manager

        Args:
            engine: SQLAlchemy engine instance
        """
        self.engine = engine
        self.migrations_dir = Path(__file__).parent
        self.available_migrations = self._discover_migrations()

    def _discover_migrations(self) -> List[str]:
        """Discover all migration files in the migrations directory"""
        migrations = []
        for file in sorted(self.migrations_dir.glob("*.py")):
            if file.name != "__init__.py" and not file.name.startswith("_"):
                migration_name = file.stem
                migrations.append(migration_name)

        logger.debug(f"Discovered {len(migrations)} migrations: {migrations}")
        return migrations

    def _load_migration(self, migration_name: str):
        """Load a migration module"""
        try:
            module = importlib.import_module(
                f"memori.database.migrations.{migration_name}"
            )
            return module
        except ImportError as e:
            logger.error(f"Failed to load migration {migration_name}: {e}")
            raise

    def _create_migration_tracking_table(self):
        """Create table to track applied migrations"""
        dialect = self.engine.dialect.name

        with self.engine.connect() as conn:
            if dialect == "sqlite":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        description TEXT,
                        applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
            elif dialect == "mysql":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(255) PRIMARY KEY,
                        description TEXT,
                        applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
            elif dialect == "postgresql":
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(255) PRIMARY KEY,
                        description TEXT,
                        applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

            conn.commit()
            logger.debug("Migration tracking table ready")

    def _get_applied_migrations(self) -> List[str]:
        """Get list of already applied migrations"""
        self._create_migration_tracking_table()

        with self.engine.connect() as conn:
            result = conn.execute("SELECT version FROM schema_migrations ORDER BY version")
            return [row[0] for row in result]

    def _mark_migration_applied(self, version: str, description: str):
        """Mark a migration as applied"""
        with self.engine.connect() as conn:
            if self.engine.dialect.name == "sqlite":
                conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations (version, description) VALUES (?, ?)",
                    (version, description),
                )
            elif self.engine.dialect.name == "mysql":
                conn.execute(
                    "INSERT IGNORE INTO schema_migrations (version, description) VALUES (%s, %s)",
                    (version, description),
                )
            elif self.engine.dialect.name == "postgresql":
                conn.execute(
                    "INSERT INTO schema_migrations (version, description) VALUES (%s, %s) ON CONFLICT (version) DO NOTHING",
                    (version, description),
                )

            conn.commit()

    def _unmark_migration(self, version: str):
        """Remove migration from applied list"""
        with self.engine.connect() as conn:
            conn.execute(
                "DELETE FROM schema_migrations WHERE version = ?", (version,)
            )
            conn.commit()

    def pending_migrations(self) -> List[str]:
        """Get list of pending (not yet applied) migrations"""
        applied = self._get_applied_migrations()
        return [m for m in self.available_migrations if m not in applied]

    def upgrade(self, target: str = None):
        """
        Apply pending migrations

        Args:
            target: Optional specific migration to upgrade to (defaults to latest)
        """
        pending = self.pending_migrations()

        if not pending:
            logger.info("No pending migrations")
            return

        if target:
            # Only apply migrations up to target
            pending = [m for m in pending if m <= target]

        logger.info(f"Applying {len(pending)} migration(s)")

        for migration_name in pending:
            logger.info(f"Applying migration: {migration_name}")
            module = self._load_migration(migration_name)

            try:
                module.upgrade(self.engine)
                version = module.get_version()
                description = module.get_description()
                self._mark_migration_applied(version, description)
                logger.success(f"✓ {migration_name} applied")

            except Exception as e:
                logger.error(f"✗ {migration_name} failed: {e}")
                raise

        logger.success("All migrations applied successfully")

    def downgrade(self, target: str):
        """
        Rollback migrations to a specific version

        Args:
            target: Migration version to rollback to
        """
        applied = self._get_applied_migrations()
        to_revert = [m for m in reversed(applied) if m > target]

        if not to_revert:
            logger.info("No migrations to revert")
            return

        logger.warning(f"Reverting {len(to_revert)} migration(s)")

        for migration_name in to_revert:
            logger.info(f"Reverting migration: {migration_name}")
            module = self._load_migration(migration_name)

            try:
                module.downgrade(self.engine)
                version = module.get_version()
                self._unmark_migration(version)
                logger.success(f"✓ {migration_name} reverted")

            except Exception as e:
                logger.error(f"✗ {migration_name} revert failed: {e}")
                raise

        logger.success("Migration rollback completed")

    def status(self):
        """Print migration status"""
        applied = self._get_applied_migrations()
        pending = self.pending_migrations()

        logger.info("=== Migration Status ===")
        logger.info(f"Database: {self.engine.dialect.name}")

        if applied:
            logger.info(f"\n✓ Applied migrations ({len(applied)}):")
            for migration in applied:
                logger.info(f"  - {migration}")

        if pending:
            logger.warning(f"\n⚠ Pending migrations ({len(pending)}):")
            for migration in pending:
                logger.warning(f"  - {migration}")
        else:
            logger.success("\n✓ Database is up to date")


def run_migrations(engine):
    """
    Convenience function to run all pending migrations

    Args:
        engine: SQLAlchemy engine instance
    """
    manager = MigrationManager(engine)
    manager.upgrade()


def get_migration_status(engine):
    """
    Get migration status for a database

    Args:
        engine: SQLAlchemy engine instance

    Returns:
        dict: Migration status information
    """
    manager = MigrationManager(engine)
    applied = manager._get_applied_migrations()
    pending = manager.pending_migrations()

    return {
        "applied": applied,
        "pending": pending,
        "database": engine.dialect.name,
        "up_to_date": len(pending) == 0,
    }
