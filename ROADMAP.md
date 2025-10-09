# Memori OSS Roadmap 2025

This roadmap shows planned and in-progress improvements for **Memori OSS.**

Community contributions are welcome. Please check the roadmap and open issues in the repo. 

## Core Features

| Feature | Description | Status | Notes | Issue Link |
| --- | --- | --- | --- | --- |
| **Ingest Unstructured Data** | Support ingestion from raw text, documents, and URLs to expand input sources. | 🟡 Planned | Enables ingestion from multiple unstructured sources. |  |
| **Graph-Based Search in SQL** | Enable hybrid relational + graph queries for contextual recall. | 🟡 In Progress | Core for semantic + relational search. |  |
| **Support for Pydantic-AI Framework** | Add native support for Pydantic-AI integration. | 🟡 Planned | Smooth integration with typed AI models. |  |
| **Add `user_id` Namespace Feature** | Allow multi-user memory isolation using namespaces. | 🟠 Buggy / Needs Fix | Implemented but has issues; debugging ongoing. |  |
| **Data Ingestion from Gibson DB** | Direct ingestion connector from GibsonAI databases. | 🟡 Planned | Needed for GibsonAI-SaaS sync. |  |
| **Image Processing in Memori** | Enable image-based retrieval with multi-turn memory. | 🟡 Planned | Use case: “Show me red shoes → under $100”. |  |
| **Methods to Connect with GibsonAI** | Improve linking between Memori OSS and GibsonAI agent infrastructure. | 🟡 Planned | Define standard connection methods. |  |
| **AzureOpenAI Auto-Record** | Auto-record short-term memory from Azure OpenAI sessions. | 🟡 Planned | Enables automatic session memory capture. |  |
| **Update `memori_schema` for GibsonAI Deployment** | Align schema with GibsonAI SaaS structure. | 🟡 Planned | Required for compatibility. |  |

## Developer Experience & Integrations

| Feature | Description | Status | Notes | Issue Link |
| --- | --- | --- | --- | --- |
| Memori REST API | First-class REST interface mirroring Python SDK | 🟡 Planned | Implement Fast API Ship OpenAPI spec + examples.  |  |
| **Update Docs** | Refresh documentation with new APIs, architecture, and examples. | 🟡 Planned | High priority for OSS visibility. |  |
| **Technical Paper of Memori** | Publish a public technical paper describing architecture and benchmarks. | 🟡 In Progress | Draft under review. |  |
| **LoCoMo Benchmark of Memori** | Benchmark Memori’s latency and recall performance. | 🟡 Planned | Compare against existing memory solutions. |  |
| **Refactor Codebase** | Clean up and modularize code for better maintainability. | 🟡 Planned | Prep for wider community contributions. |  |
| **Improve Error Handling (DB Dependency)** | Add graceful fallbacks for database and schema dependency issues. | 🟡 In Progress | Improves reliability across deployments. |  |

## Stability, Testing & Bug Fixes

| Feature | Description | Status | Notes | Issue Link |
| --- | --- | --- | --- | --- |
| **Duplicate Memory Creation** | Fix duplicate entries appearing in both short-term and long-term memory. | 🟠 Known Issue | Observed during testing. |  |
| **Search Recursion Issue** | Resolve recursive memory lookups in remote DB environments. | 🔴 Critical | High-priority fix needed. |  |
| **Postgres FTS (Neon) Issue** | Fix partial search failure with full-text search on Neon Postgres. | 🟡 Known Issue | Search works partially but inconsistently. |  |
| **Gibson Issues with Memori** | Debug integration-level issues when used within GibsonAI. | 🟡 Planned | Needs collaboration with GibsonAI team. |  |