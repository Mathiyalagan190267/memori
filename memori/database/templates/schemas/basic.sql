-- Memori v2.0 Streamlined Database Schema
-- Simplified schema with only essential tables for production use

-- Chat History Table
-- Stores all conversations between users and AI systems
CREATE TABLE IF NOT EXISTS chat_history (
    chat_id TEXT PRIMARY KEY,
    user_input TEXT NOT NULL,
    ai_output TEXT NOT NULL,
    model TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    session_id TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    tokens_used INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

-- Short-term Memory Table (with full ProcessedMemory structure)
-- Stores temporary memories with expiration (auto-expires after ~7 days)
-- Also stores permanent user context when expires_at is NULL
CREATE TABLE IF NOT EXISTS short_term_memory (
    memory_id TEXT PRIMARY KEY,
    chat_id TEXT,
    processed_data TEXT NOT NULL,  -- Full ProcessedMemory JSON
    importance_score REAL NOT NULL DEFAULT 0.5,
    category_primary TEXT NOT NULL,  -- Extracted for indexing
    retention_type TEXT NOT NULL DEFAULT 'short_term',
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,  -- NULL = permanent storage (for user context)
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    searchable_content TEXT NOT NULL,  -- Optimized for search
    summary TEXT NOT NULL,  -- Concise summary
    is_permanent_context BOOLEAN DEFAULT 0,  -- Marks permanent user context
    FOREIGN KEY (chat_id) REFERENCES chat_history (chat_id)
);

-- Long-term Memory Table (Enhanced with Classification and Conscious Context)
-- Stores persistent memories with intelligent classification and deduplication
CREATE TABLE IF NOT EXISTS long_term_memory (
    memory_id TEXT PRIMARY KEY,
    original_chat_id TEXT,
    processed_data TEXT NOT NULL,  -- Full ProcessedLongTermMemory JSON
    importance_score REAL NOT NULL DEFAULT 0.5,
    category_primary TEXT NOT NULL,  -- Extracted for indexing
    retention_type TEXT NOT NULL DEFAULT 'long_term',
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    searchable_content TEXT NOT NULL,  -- Optimized for search
    summary TEXT NOT NULL,  -- Concise summary
    novelty_score REAL DEFAULT 0.5,
    relevance_score REAL DEFAULT 0.5,
    actionability_score REAL DEFAULT 0.5,
    
    -- Enhanced Classification Fields
    classification TEXT NOT NULL DEFAULT 'conversational',  -- essential, contextual, conversational, reference, personal, conscious-info
    memory_importance TEXT NOT NULL DEFAULT 'medium',  -- critical, high, medium, low
    topic TEXT,  -- Main topic/subject
    entities_json TEXT DEFAULT '[]',  -- JSON array of extracted entities
    keywords_json TEXT DEFAULT '[]',  -- JSON array of keywords for search
    
    -- Conscious Context Flags
    is_user_context BOOLEAN DEFAULT 0,  -- Contains user personal info
    is_preference BOOLEAN DEFAULT 0,    -- User preference/opinion
    is_skill_knowledge BOOLEAN DEFAULT 0,  -- User abilities/expertise
    is_current_project BOOLEAN DEFAULT 0,  -- Current work context
    promotion_eligible BOOLEAN DEFAULT 0,  -- Should be promoted to short-term
    
    -- Memory Management
    duplicate_of TEXT,  -- Links to original if duplicate
    supersedes_json TEXT DEFAULT '[]',  -- JSON array of memory IDs this replaces
    related_memories_json TEXT DEFAULT '[]',  -- JSON array of connected memory IDs
    
    -- Technical Metadata
    confidence_score REAL DEFAULT 0.8,  -- AI confidence in extraction
    extraction_timestamp TIMESTAMP NOT NULL,
    classification_reason TEXT,  -- Why this classification was chosen
    
    -- Processing Status
    processed_for_duplicates BOOLEAN DEFAULT 0,  -- Processed for duplicate detection
    conscious_processed BOOLEAN DEFAULT 0  -- Processed for conscious context extraction
);

-- ======================================
-- GRAPH-BASED SEARCH TABLES
-- ======================================

-- Memory Entities Table
-- Stores extracted entities for graph-based search and relationship building
CREATE TABLE IF NOT EXISTS memory_entities (
    entity_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,  -- 'short_term' or 'long_term'
    entity_type TEXT NOT NULL,  -- person, technology, topic, skill, project, keyword
    entity_value TEXT NOT NULL,
    normalized_value TEXT NOT NULL,  -- Lowercase for case-insensitive matching
    relevance_score REAL DEFAULT 0.5,
    namespace TEXT NOT NULL DEFAULT 'default',
    frequency INTEGER DEFAULT 1,  -- How many times mentioned
    created_at TIMESTAMP NOT NULL,
    context TEXT  -- Additional context about this entity
);

-- Memory Relationships Table
-- Stores graph relationships between memories for advanced traversal
CREATE TABLE IF NOT EXISTS memory_relationships (
    relationship_id TEXT PRIMARY KEY,
    source_memory_id TEXT NOT NULL,
    target_memory_id TEXT NOT NULL,
    source_memory_type TEXT NOT NULL,  -- 'short_term' or 'long_term'
    target_memory_type TEXT NOT NULL,
    relationship_type TEXT NOT NULL,  -- semantic_similarity, causality, reference, etc.
    strength REAL NOT NULL DEFAULT 0.5,  -- 0.0-1.0
    bidirectional BOOLEAN DEFAULT 1,
    namespace TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMP NOT NULL,
    last_strengthened TIMESTAMP,  -- When strength was last updated
    access_count INTEGER DEFAULT 0,  -- How often traversed
    reasoning TEXT,  -- Why this relationship exists
    shared_entity_count INTEGER DEFAULT 0,  -- Number of shared entities
    metadata_json TEXT DEFAULT '{}'  -- Additional relationship metadata
);

-- Performance Indexes

-- Chat History Indexes
CREATE INDEX IF NOT EXISTS idx_chat_namespace_session ON chat_history(namespace, session_id);
CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_chat_model ON chat_history(model);

-- Short-term Memory Indexes
CREATE INDEX IF NOT EXISTS idx_short_term_namespace ON short_term_memory(namespace);
CREATE INDEX IF NOT EXISTS idx_short_term_category ON short_term_memory(category_primary);
CREATE INDEX IF NOT EXISTS idx_short_term_importance ON short_term_memory(importance_score);
CREATE INDEX IF NOT EXISTS idx_short_term_expires ON short_term_memory(expires_at);
CREATE INDEX IF NOT EXISTS idx_short_term_created ON short_term_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_short_term_searchable ON short_term_memory(searchable_content);
CREATE INDEX IF NOT EXISTS idx_short_term_access ON short_term_memory(access_count, last_accessed);
CREATE INDEX IF NOT EXISTS idx_short_term_permanent ON short_term_memory(is_permanent_context);

-- Long-term Memory Indexes  
CREATE INDEX IF NOT EXISTS idx_long_term_namespace ON long_term_memory(namespace);
CREATE INDEX IF NOT EXISTS idx_long_term_category ON long_term_memory(category_primary);
CREATE INDEX IF NOT EXISTS idx_long_term_importance ON long_term_memory(importance_score);
CREATE INDEX IF NOT EXISTS idx_long_term_created ON long_term_memory(created_at);
CREATE INDEX IF NOT EXISTS idx_long_term_searchable ON long_term_memory(searchable_content);
CREATE INDEX IF NOT EXISTS idx_long_term_access ON long_term_memory(access_count, last_accessed);
CREATE INDEX IF NOT EXISTS idx_long_term_scores ON long_term_memory(novelty_score, relevance_score, actionability_score);

-- Enhanced Classification Indexes
CREATE INDEX IF NOT EXISTS idx_long_term_classification ON long_term_memory(classification);
CREATE INDEX IF NOT EXISTS idx_long_term_memory_importance ON long_term_memory(memory_importance);
CREATE INDEX IF NOT EXISTS idx_long_term_topic ON long_term_memory(topic);
CREATE INDEX IF NOT EXISTS idx_long_term_conscious_flags ON long_term_memory(is_user_context, is_preference, is_skill_knowledge, promotion_eligible);
CREATE INDEX IF NOT EXISTS idx_long_term_conscious_processed ON long_term_memory(conscious_processed);
CREATE INDEX IF NOT EXISTS idx_long_term_duplicates ON long_term_memory(processed_for_duplicates);
CREATE INDEX IF NOT EXISTS idx_long_term_confidence ON long_term_memory(confidence_score);

-- Graph Entity Indexes (9 indexes for fast entity search)
CREATE INDEX IF NOT EXISTS idx_entity_memory ON memory_entities(memory_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_entity_type ON memory_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_value ON memory_entities(entity_value);
CREATE INDEX IF NOT EXISTS idx_entity_normalized ON memory_entities(normalized_value);
CREATE INDEX IF NOT EXISTS idx_entity_namespace ON memory_entities(namespace);
CREATE INDEX IF NOT EXISTS idx_entity_relevance ON memory_entities(relevance_score);
CREATE INDEX IF NOT EXISTS idx_entity_type_value ON memory_entities(entity_type, normalized_value);
CREATE INDEX IF NOT EXISTS idx_entity_namespace_type ON memory_entities(namespace, entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_compound ON memory_entities(namespace, entity_type, normalized_value, relevance_score);

-- Graph Relationship Indexes (12 indexes for fast graph traversal)
CREATE INDEX IF NOT EXISTS idx_rel_source ON memory_relationships(source_memory_id, source_memory_type);
CREATE INDEX IF NOT EXISTS idx_rel_target ON memory_relationships(target_memory_id, target_memory_type);
CREATE INDEX IF NOT EXISTS idx_rel_type ON memory_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_rel_strength ON memory_relationships(strength);
CREATE INDEX IF NOT EXISTS idx_rel_namespace ON memory_relationships(namespace);
CREATE INDEX IF NOT EXISTS idx_rel_bidirectional ON memory_relationships(bidirectional);
CREATE INDEX IF NOT EXISTS idx_rel_source_type ON memory_relationships(source_memory_id, relationship_type);
CREATE INDEX IF NOT EXISTS idx_rel_target_type ON memory_relationships(target_memory_id, relationship_type);
CREATE INDEX IF NOT EXISTS idx_rel_compound_source ON memory_relationships(source_memory_id, relationship_type, strength);
CREATE INDEX IF NOT EXISTS idx_rel_compound_target ON memory_relationships(target_memory_id, relationship_type, strength);
CREATE INDEX IF NOT EXISTS idx_rel_namespace_type ON memory_relationships(namespace, relationship_type, strength);
CREATE INDEX IF NOT EXISTS idx_rel_entity_count ON memory_relationships(shared_entity_count);

-- Full-Text Search Support (SQLite FTS5)
-- Enables advanced text search capabilities
CREATE VIRTUAL TABLE IF NOT EXISTS memory_search_fts USING fts5(
    memory_id,
    memory_type,
    namespace,
    searchable_content,
    summary,
    category_primary,
    content='',
    contentless_delete=1
);

-- Triggers to maintain FTS index
CREATE TRIGGER IF NOT EXISTS short_term_memory_fts_insert AFTER INSERT ON short_term_memory
BEGIN
    INSERT INTO memory_search_fts(memory_id, memory_type, namespace, searchable_content, summary, category_primary)
    VALUES (NEW.memory_id, 'short_term', NEW.namespace, NEW.searchable_content, NEW.summary, NEW.category_primary);
END;

CREATE TRIGGER IF NOT EXISTS long_term_memory_fts_insert AFTER INSERT ON long_term_memory
BEGIN
    INSERT INTO memory_search_fts(memory_id, memory_type, namespace, searchable_content, summary, category_primary)
    VALUES (NEW.memory_id, 'long_term', NEW.namespace, NEW.searchable_content, NEW.summary, NEW.category_primary);
END;

CREATE TRIGGER IF NOT EXISTS short_term_memory_fts_delete AFTER DELETE ON short_term_memory
BEGIN
    DELETE FROM memory_search_fts WHERE memory_id = OLD.memory_id AND memory_type = 'short_term';
END;

CREATE TRIGGER IF NOT EXISTS long_term_memory_fts_delete AFTER DELETE ON long_term_memory
BEGIN
    DELETE FROM memory_search_fts WHERE memory_id = OLD.memory_id AND memory_type = 'long_term';
END;
