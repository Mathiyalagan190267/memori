"""
MongoDB connector for Memori
Provides MongoDB-specific implementation of the database connector interface
"""

import json
from typing import Any
from urllib.parse import urlparse

from loguru import logger

try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
    from pymongo.errors import ConnectionFailure, OperationFailure

    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

from ...utils.exceptions import DatabaseError
from .base_connector import BaseDatabaseConnector, DatabaseType


class MongoDBConnector(BaseDatabaseConnector):
    """MongoDB database connector with Atlas Vector Search support"""

    def __init__(self, connection_config):
        """Initialize MongoDB connector"""
        if not PYMONGO_AVAILABLE:
            raise DatabaseError(
                "pymongo is required for MongoDB support. Install with: pip install pymongo"
            )

        if isinstance(connection_config, str):
            self.connection_string = connection_config
            self.connection_config = {"connection_string": connection_config}
        else:
            self.connection_string = connection_config.get(
                "connection_string", "mongodb://localhost:27017"
            )

        # Parse MongoDB connection string
        self._parse_connection_string()

        # MongoDB-specific settings
        self.client = None
        self.database = None
        self._collections = {}

        super().__init__(connection_config)

    def _detect_database_type(self) -> DatabaseType:
        """Detect database type from connection config"""
        return DatabaseType.MONGODB

    def _parse_connection_string(self):
        """Parse MongoDB connection string to extract components"""
        try:
            parsed = urlparse(self.connection_string)
            self.host = parsed.hostname or "localhost"
            self.port = parsed.port or 27017
            self.database_name = parsed.path.lstrip("/") or "memori"
            self.username = parsed.username
            self.password = parsed.password

            # Extract query parameters
            self.options = {}
            if parsed.query:
                params = parsed.query.split("&")
                for param in params:
                    if "=" in param:
                        key, value = param.split("=", 1)
                        self.options[key] = value

        except Exception as e:
            logger.warning(f"Failed to parse MongoDB connection string: {e}")
            # Set defaults
            self.host = "localhost"
            self.port = 27017
            self.database_name = "memori"
            self.username = None
            self.password = None
            self.options = {}

    def get_connection(self) -> MongoClient:
        """Get MongoDB client connection"""
        if self.client is None:
            try:
                # Create MongoDB client with appropriate options
                client_options = {
                    "serverSelectionTimeoutMS": 5000,  # 5 second timeout
                    "connectTimeoutMS": 10000,  # 10 second connect timeout
                    "socketTimeoutMS": 30000,  # 30 second socket timeout
                    "maxPoolSize": 50,  # Connection pool size
                    "retryWrites": True,  # Enable retryable writes
                }

                # Add any additional options from connection string
                client_options.update(self.options)

                self.client = MongoClient(self.connection_string, **client_options)

                # Test connection
                self.client.admin.command("ping")
                logger.info(f"Connected to MongoDB at {self.host}:{self.port}")

            except Exception as e:
                raise DatabaseError(f"Failed to connect to MongoDB: {e}")

        return self.client

    def get_database(self) -> Database:
        """Get MongoDB database"""
        if self.database is None:
            client = self.get_connection()
            self.database = client[self.database_name]
        return self.database

    def get_collection(self, collection_name: str) -> Collection:
        """Get MongoDB collection with caching"""
        if collection_name not in self._collections:
            database = self.get_database()
            self._collections[collection_name] = database[collection_name]
        return self._collections[collection_name]

    def execute_query(
        self, query: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a query-like operation in MongoDB
        Note: MongoDB doesn't use SQL, so this is adapted for MongoDB operations
        """
        try:
            # Parse the "query" as a JSON operation for MongoDB
            # This is a compatibility layer for the base interface
            if isinstance(query, str) and query.strip().startswith("{"):
                # Treat as MongoDB operation
                operation = json.loads(query)
                collection_name = operation.get("collection", "memories")
                operation_type = operation.get("operation", "find")
                filter_doc = operation.get("filter", {})
                options = operation.get("options", {})

                collection = self.get_collection(collection_name)

                if operation_type == "find":
                    cursor = collection.find(filter_doc, **options)
                    results = list(cursor)
                    # Convert ObjectId to string for JSON serialization
                    for result in results:
                        if "_id" in result:
                            result["_id"] = str(result["_id"])
                    return results
                elif operation_type == "aggregate":
                    pipeline = operation.get("pipeline", [])
                    cursor = collection.aggregate(pipeline, **options)
                    results = list(cursor)
                    # Convert ObjectId to string for JSON serialization
                    for result in results:
                        if "_id" in result:
                            result["_id"] = str(result["_id"])
                    return results
                else:
                    raise DatabaseError(
                        f"Unsupported MongoDB operation: {operation_type}"
                    )
            else:
                # Fallback: treat as a collection name and return all documents
                collection = self.get_collection(query or "memories")
                cursor = collection.find().limit(100)  # Limit for safety
                results = list(cursor)
                # Convert ObjectId to string for JSON serialization
                for result in results:
                    if "_id" in result:
                        result["_id"] = str(result["_id"])
                return results

        except Exception as e:
            raise DatabaseError(f"Failed to execute MongoDB query: {e}")

    def execute_insert(self, query: str, params: list[Any] | None = None) -> str:
        """Execute an insert operation and return the inserted document ID"""
        try:
            if isinstance(query, str) and query.strip().startswith("{"):
                # Parse as MongoDB insert operation
                operation = json.loads(query)
                collection_name = operation.get("collection", "memories")
                document = operation.get("document", {})

                collection = self.get_collection(collection_name)
                result = collection.insert_one(document)
                return str(result.inserted_id)
            else:
                raise DatabaseError("Invalid insert operation format for MongoDB")

        except Exception as e:
            raise DatabaseError(f"Failed to execute MongoDB insert: {e}")

    def execute_update(self, query: str, params: list[Any] | None = None) -> int:
        """Execute an update operation and return number of modified documents"""
        try:
            if isinstance(query, str) and query.strip().startswith("{"):
                # Parse as MongoDB update operation
                operation = json.loads(query)
                collection_name = operation.get("collection", "memories")
                filter_doc = operation.get("filter", {})
                update_doc = operation.get("update", {})
                options = operation.get("options", {})

                collection = self.get_collection(collection_name)

                if operation.get("update_many", False):
                    result = collection.update_many(filter_doc, update_doc, **options)
                else:
                    result = collection.update_one(filter_doc, update_doc, **options)

                return result.modified_count
            else:
                raise DatabaseError("Invalid update operation format for MongoDB")

        except Exception as e:
            raise DatabaseError(f"Failed to execute MongoDB update: {e}")

    def execute_delete(self, query: str, params: list[Any] | None = None) -> int:
        """Execute a delete operation and return number of deleted documents"""
        try:
            if isinstance(query, str) and query.strip().startswith("{"):
                # Parse as MongoDB delete operation
                operation = json.loads(query)
                collection_name = operation.get("collection", "memories")
                filter_doc = operation.get("filter", {})
                options = operation.get("options", {})

                collection = self.get_collection(collection_name)

                if operation.get("delete_many", False):
                    result = collection.delete_many(filter_doc, **options)
                else:
                    result = collection.delete_one(filter_doc, **options)

                return result.deleted_count
            else:
                raise DatabaseError("Invalid delete operation format for MongoDB")

        except Exception as e:
            raise DatabaseError(f"Failed to execute MongoDB delete: {e}")

    def execute_transaction(self, queries: list[tuple[str, list[Any] | None]]) -> bool:
        """Execute multiple operations in a MongoDB transaction"""
        try:
            client = self.get_connection()

            # Check if transactions are supported (requires replica set or sharded cluster)
            try:
                with client.start_session() as session:
                    with session.start_transaction():
                        for query, params in queries:
                            # Execute each operation within the transaction
                            if "insert" in query.lower():
                                self.execute_insert(query, params)
                            elif "update" in query.lower():
                                self.execute_update(query, params)
                            elif "delete" in query.lower():
                                self.execute_delete(query, params)

                        # Transaction commits automatically if no exception is raised
                        return True

            except OperationFailure as e:
                if "Transaction numbers" in str(e):
                    # Transactions not supported, execute operations individually
                    logger.warning(
                        "Transactions not supported, executing operations individually"
                    )
                    for query, params in queries:
                        if "insert" in query.lower():
                            self.execute_insert(query, params)
                        elif "update" in query.lower():
                            self.execute_update(query, params)
                        elif "delete" in query.lower():
                            self.execute_delete(query, params)
                    return True
                else:
                    raise

        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            return False

    def test_connection(self) -> bool:
        """Test if the MongoDB connection is working"""
        try:
            client = self.get_connection()
            # Ping the server
            client.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False

    def initialize_schema(self, schema_sql: str | None = None):
        """Initialize MongoDB collections and indexes"""
        try:
            from ..schema_generators.mongodb_schema_generator import (
                MongoDBSchemaGenerator,
            )

            schema_generator = MongoDBSchemaGenerator()
            database = self.get_database()

            # Create collections with validation rules
            collections_schema = schema_generator.generate_collections_schema()
            for collection_name, schema in collections_schema.items():
                if collection_name not in database.list_collection_names():
                    # Create collection with validation
                    database.create_collection(
                        collection_name,
                        validator=schema.get("validator"),
                        validationAction=schema.get("validationAction", "error"),
                        validationLevel=schema.get("validationLevel", "strict"),
                    )
                    logger.info(f"Created MongoDB collection: {collection_name}")

            # Create indexes
            indexes_schema = schema_generator.generate_indexes_schema()
            for collection_name, indexes in indexes_schema.items():
                collection = self.get_collection(collection_name)
                for index in indexes:
                    try:
                        collection.create_index(
                            index["keys"],
                            name=index.get("name"),
                            unique=index.get("unique", False),
                            sparse=index.get("sparse", False),
                            background=True,  # Create index in background
                        )
                        logger.debug(f"Created index on {collection_name}: {index}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to create index on {collection_name}: {e}"
                        )

            logger.info("MongoDB schema initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize MongoDB schema: {e}")
            raise DatabaseError(f"Failed to initialize MongoDB schema: {e}")

    def supports_full_text_search(self) -> bool:
        """Check if MongoDB supports text search (always True for MongoDB)"""
        return True

    def supports_vector_search(self) -> bool:
        """Check if MongoDB Atlas Vector Search is available"""
        try:
            # Check if this is MongoDB Atlas by looking for Atlas-specific features
            client = self.get_connection()
            build_info = client.admin.command("buildInfo")

            # Atlas typically includes specific modules or version patterns
            # This is a heuristic check - in production you might want to configure this explicitly
            server_version = build_info.get("version", "")
            modules = build_info.get("modules", [])

            # Check if vector search is available (Atlas feature)
            # This is a simplified check - Atlas vector search availability can be complex
            return "atlas" in str(modules).lower() or self._is_atlas_connection()

        except Exception:
            return False

    def _is_atlas_connection(self) -> bool:
        """Heuristic to detect if this is an Atlas connection"""
        return (
            "mongodb.net" in self.connection_string.lower()
            or "atlas" in self.connection_string.lower()
            or "cluster" in self.connection_string.lower()
        )

    def create_full_text_index(
        self, table: str, columns: list[str], index_name: str
    ) -> str:
        """Create MongoDB text index"""
        try:
            collection = self.get_collection(table)

            # Create text index specification
            index_spec = {}
            for column in columns:
                index_spec[column] = "text"

            collection.create_index(
                list(index_spec.items()), name=index_name, background=True
            )

            return f"Created text index '{index_name}' on collection '{table}'"

        except Exception as e:
            raise DatabaseError(f"Failed to create text index: {e}")

    def create_vector_index(
        self,
        collection_name: str,
        vector_field: str,
        dimensions: int,
        similarity: str = "cosine",
        index_name: str | None = None,
    ) -> str:
        """Create MongoDB Atlas Vector Search index"""
        try:
            if not self.supports_vector_search():
                raise DatabaseError(
                    "Vector search is not supported in this MongoDB deployment"
                )

            collection = self.get_collection(collection_name)

            # Vector search index specification for MongoDB Atlas
            vector_index_spec = {
                "fields": [
                    {
                        "path": vector_field,
                        "type": "vector",
                        "similarity": similarity,
                        "dimensions": dimensions,
                    }
                ]
            }

            index_name = index_name or f"{vector_field}_vector_index"

            # Note: Vector search indexes are typically created via Atlas UI or Atlas Admin API
            # This is a placeholder for the actual implementation
            logger.warning(
                "Vector search indexes should be created via MongoDB Atlas UI or Admin API"
            )

            return f"Vector index specification created for '{collection_name}.{vector_field}'"

        except Exception as e:
            raise DatabaseError(f"Failed to create vector index: {e}")

    def get_database_info(self) -> dict[str, Any]:
        """Get MongoDB database information and capabilities"""
        try:
            client = self.get_connection()
            database = self.get_database()

            info = {}

            # Server information
            server_info = client.server_info()
            info["version"] = server_info.get("version", "unknown")
            info["database_type"] = self.database_type.value
            info["database_name"] = self.database_name
            info["connection_string"] = (
                self.connection_string.replace(
                    f"{self.username}:{self.password}@", "***:***@"
                )
                if self.username and self.password
                else self.connection_string
            )

            # Database stats
            try:
                stats = database.command("dbStats")
                info["collections_count"] = stats.get("collections", 0)
                info["data_size"] = stats.get("dataSize", 0)
                info["storage_size"] = stats.get("storageSize", 0)
                info["indexes_count"] = stats.get("indexes", 0)
            except Exception:
                pass

            # Capabilities
            info["full_text_search_support"] = True
            info["vector_search_support"] = self.supports_vector_search()
            info["transactions_support"] = self._check_transactions_support()

            # Replica set information
            try:
                replica_config = client.admin.command("replSetGetStatus")
                info["replica_set"] = replica_config.get("set", "Not in replica set")
            except Exception:
                info["replica_set"] = "Standalone"

            return info

        except Exception as e:
            logger.warning(f"Could not get MongoDB database info: {e}")
            return {
                "database_type": self.database_type.value,
                "version": "unknown",
                "full_text_search_support": True,
                "vector_search_support": False,
                "error": str(e),
            }

    def _check_transactions_support(self) -> bool:
        """Check if MongoDB deployment supports transactions"""
        try:
            client = self.get_connection()
            with client.start_session() as session:
                with session.start_transaction():
                    # Just test if we can start a transaction
                    pass
            return True
        except Exception:
            return False

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            self._collections.clear()
            logger.info("MongoDB connection closed")

    def __del__(self):
        """Cleanup when connector is destroyed"""
        self.close()
