# RAG using LlamaIndex

A Retrieval Augmented Generation (RAG) system that uses sub-question and tool-selecting query engines from LlamaIndex to query Paul Graham's article "What I Worked On".

## Quick Start

### Requirements

- Python 3.11.1+ (matches `pyproject.toml`)
- Access to Milvus 2.x at `http://localhost:19530` and MongoDB at `mongodb://localhost:27017/` (default URIs hard-coded in `langextract_simple.py`)

### Clone the Repository

```bash
git clone https://github.com/tonypeng1/llamaindex.git
cd llamaindex
```

### Install Dependencies with uv (Recommended)

This project uses [`uv`](https://github.com/astral-sh/uv) for fast Python dependency management.

Install `uv` if you haven't already:
```bash
pip install uv
```

Install project dependencies:
```bash
uv pip install -e .
```

> **Note:** You can still use regular `pip install -e .` if preferred, but `uv` is recommended for speed and reliability.

### Setup

1. **Download the article**: Get Paul Graham's "What I Worked On" ([download here](https://drive.google.com/file/d/1YzCscCmQXn2IcGS-omcAc8TBuFrpiN4-/view?usp=sharing)) and place it at:
   ```
   ./data/paul_graham/paul_graham_essay.pdf
   ```

2. **Database setup**: Ensure Milvus and MongoDB are running locally or accessible via network. By default the script points to `http://localhost:19530` for Milvus and `mongodb://localhost:27017/` for MongoDB—change `uri_milvus` / `uri_mongo` in `langextract_simple.py` if you host them elsewhere.

3. **API Keys**: Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

4. **Run the pipeline**: Execute the main script once the databases and keys are ready:
    ```bash
    uv run python langextract_simple.py
    # or just: python langextract_simple.py
    ```
    Customize `metadata`, `schema_name`, and other knobs in the script before running if you want a different extraction mode.

### Contributing

Please use `uv` for installing dependencies and managing your Python environment.

---

## Key Features

- **Dual Storage**: Uses Milvus (vector database) and MongoDB (document store) with duplicate prevention
- **Multi-Tool Query Engine**: Three specialized tools:
  - `keyphrase_tool`: Answers specific factual queries
  - `page_filter_tool`: Retrieves information from specific pages
  - `summary_tool`: Provides full-context summaries
- **Flexible Metadata Extraction**: Four extraction options:
  - **None**: Basic chunking only (fastest, free)
  - **EntityExtractor**: Named entity recognition (fast, free)
  - **LangExtract**: Rich semantic metadata extraction (slow, paid, comprehensive)
  - **Both**: EntityExtractor + LangExtract (maximum metadata richness)
- **Entity-Based Filtering**: Smart query filtering using extracted entity metadata
  - Automatically detects entities in queries (people, organizations, locations)
  - Filters **vector retriever only** to nodes mentioning those entities
  - BM25 retriever operates on full docstore (no entity filtering)
  - Improves precision by 40-60% for entity-focused queries
- **Hybrid Retrieval**: Combines vector similarity search with BM25 keyword matching
  - **BM25 Retriever**: Operates on MongoDB docstore, keyword-based, no entity filtering
  - **Vector Retriever**: Operates on Milvus vector index, semantic search, optional entity filtering
    - **Fusion**: Uses `QueryFusionRetriever` in reciprocal rank fusion mode to merge both rankings (no manual weighting required)
- **KeyBERT Integration**: Reduces BM25 noise by extracting key phrases before retrieval
- **Advanced Post-processing**: Reranking, context expansion, and page-based sorting
- **Configurable Pipeline**: Fine-tune chunking, overlap, and retrieval parameters

## Metadata Extraction Options

The system supports four metadata extraction methods to suit different needs:

### 1. None (Basic) - Fast & Free
- Basic chunking with page numbers only
- **Use case**: Quick testing, simple documents
- **Speed**: ⚡⚡⚡ Very Fast
- **Cost**: FREE

### 2. EntityExtractor - Fast & Free
- Named entity recognition (PERSON, ORGANIZATION, LOCATION, etc.)
- Uses HuggingFace span-marker model locally
- **Use case**: Standard entity recognition needs
- **Speed**: ⚡⚡ Fast
- **Cost**: FREE

### 3. LangExtract - Slow & Paid
- Rich semantic metadata: concepts, advice, experiences, entities, time references
- Uses Google's LangExtract with OpenAI GPT-4
- **Use case**: Deep semantic understanding, complex queries
- **Speed**: ⚡ Slow
- **Cost**: LLM API cost

### 4. Both - Most Comprehensive
- Combines EntityExtractor + LangExtract
- **Use case**: Maximum metadata richness
- **Speed**: ⚡ Slowest
- **Cost**: LLM API cost

## Quick Configuration:
```python
# In langextract_simple.py, set:
metadata = "langextract"  # Options: None, "entity", "langextract", "both"
schema_name = "paul_graham_detailed"  # For LangExtract
use_entity_filtering = True  # Enable entity-based filtering (default: True)

# Advanced Configuration
chunk_size = 256              # Smaller chunks = more precise retrieval
chunk_overlap = 64            # Overlap to maintain context
similarity_top_k_fusion = 48  # Initial retrieval count (matches current script)
fusion_top_n = 42             # Post-fusion count before reranking
rerank_top_n = 32             # Nodes kept after ColBERT re-ranking
num_queries = 1               # Fusion query fan-out (1 = disable query generation)
num_nodes = 0                 # Neighbor nodes to add via SafePrevNextNodePostprocessor
```

## Schema Definitions and Dynamic Loading

The LangExtract metadata extraction uses schema definitions that specify allowed attribute values (e.g., `concept_categories`, `advice_domains`, `experience_periods`). These definitions are managed by `get_paul_graham_schema_definitions()` in `langextract_schemas.py`.

### Two Operating Modes

| Mode | When Used | Source | Purpose |
|------|-----------|--------|---------|
| **Static** (`use_dynamic_loading=False`) | During ingestion | Hardcoded defaults | Guides LLM on what attributes to extract |
| **Dynamic** (`use_dynamic_loading=True`) | During query filtering | MongoDB collection | Tells LLM what filter values actually exist |

### Why Two Modes?

1. **Ingestion Time**: When processing documents for the first time, the MongoDB collection doesn't exist yet. The schema uses static defaults to tell the extraction LLM what attribute categories are valid (e.g., "classify this concept as one of: technology, business, startups...").

2. **Query Time**: When filtering queries (e.g., "What advice about startups?"), the function loads actual distinct values from MongoDB. This ensures the LLM only suggests filters that match stored metadata.

### Key Functions

```python
# langextract_schemas.py

get_paul_graham_schema_definitions(use_dynamic_loading=True)
# Returns dict of allowed attribute values
# Example: {"concept_categories": ["technology", "startups", ...], ...}

get_paul_graham_essay_schema()
# Builds the extraction prompt + examples for LangExtract
# Uses static defaults (use_dynamic_loading=False)

get_schema(schema_name)
# Factory function to get schema by name
# Available: "paul_graham_detailed", "paul_graham_simple"
```

```python
# langextract_integration.py

extract_query_metadata_filters(query_str, schema_name)
# Analyzes user query to extract metadata filters
# Uses dynamic loading (use_dynamic_loading=True) to get actual stored values
```

### Schema Attributes

The Paul Graham detailed schema extracts these attribute categories:

| Category | Example Values | Used For |
|----------|---------------|----------|
| `concept_categories` | technology, startups, programming, life | Classifying key concepts |
| `advice_domains` | career, startups, creativity, relationships | Categorizing advice |
| `experience_periods` | childhood, college, viaweb, yc, post_yc | Timeline of experiences |
| `experience_sentiments` | positive, negative, neutral, mixed | Emotional tone |
| `entity_roles` | founder, colleague, investor, company | People/org classification |
| `time_decades` | 1970s, 1980s, 1990s, 2000s | Time period references |

**Documentation:**
- See `README_GUIDE.md` for comprehensive guide (metadata extraction, entity filtering, troubleshooting)
- See `EXAMPLES_METADATA.py` for quick-start examples and code snippets

## File Structure

Main files:
- `langextract_simple.py` - Main RAG implementation with flexible metadata extraction
- `langextract_integration.py` - LangExtract integration module
- `langextract_schemas.py` - LangExtract extraction schemas
- `utils.py` - Helper functions
- `db_operation.py` - Database operations

Documentation:
- `README_GUIDE.md` - Comprehensive guide covering metadata extraction, entity filtering, visual guides, and troubleshooting
- `EXAMPLES_METADATA.py` - Quick-start examples and configuration templates

Tests:
- `test/test_entity_filtering.py` - Entity filtering test suite
- `test/test_langextract_install.py` - LangExtract installation verification
- `test/test_langextract_schema.py` - LangExtract schema testing
- `test/demo_metadata_comparison.py` - Metadata options comparison demo

Legacy files (archived):
- `llama_bm25_simple.py` - Previous main implementation
- `utility_simple.py` - Previous helper functions
- `database_operation_simple.py` - Previous database operations

## Examples

See [this Medium article](https://medium.com/@tony3t3t/rag-with-sub-question-and-tool-selecting-query-engines-using-llamaindex-05349cb4120c) for query examples and detailed explanations.

---

## Technical Details

### Overview

`langextract_simple.py` is a RAG system combining vector-based semantic search with BM25 keyword retrieval using sub-question decomposition for PDF document Q&A.

### Core Workflow

1. **Document Processing**: Loads PDFs and splits into sentence-based chunks
2. **Dual Storage**: Embeddings in Milvus, metadata in MongoDB
3. **Multi-Tool Routing**: Routes queries to specialized tools via sub-question decomposition
4. **Re-ranking**: ColBERT neural re-ranking for improved retrieval quality

### Key Libraries

| Category | Components |
|----------|------------|
| **LlamaIndex Core** | `VectorStoreIndex`, `Settings`, `IngestionPipeline`, `SentenceSplitter`, `SubQuestionQueryEngine`, `QueryEngineTool`, `MetadataFilters` |
| **LLM & Embeddings** | Anthropic (claude-sonnet-4-0), OpenAI (text-embedding-3-small) |
| **Processors** | `EntityExtractor`, `ColbertRerank`, `PrevNextNodePostprocessor`, `PageSortNodePostprocessor` |
| **Retrieval** | `BM25Retriever`, `QueryFusionRetriever` |
| **Storage** | `PyMuPDFReader`, `MilvusVectorStore`, `MongoDocumentStore` |
| **Query Generation** | `GuidanceQuestionGenerator`, `KeyBERT` |

### Custom Modules

- **`db_operation.py`**: Database existence checks, creation, and persistence management
- **`utils.py`**: Prompt templates, fusion retrieval, tool configuration, keyphrase extraction

## Technical Architecture

**Hybrid Retrieval Pipeline:**
- **Semantic Search** (Milvus): Dense vector embeddings for conceptual similarity
- **Keyword Search** (MongoDB): BM25 for precise keyword matching
- **Fusion**: Reciprocal rank fusion (50/50 weighting)
- **Re-ranking**: ColBERT for fine-grained relevance scoring

### Entity Filtering (when enabled)

| Retriever | Database | Entity Filtering | Purpose |
|-----------|----------|------------------|---------|
| BM25 | MongoDB | None | Ensures keyword matches aren't missed |
| Vector | Milvus | Applied | Improves precision for entity-focused queries |

Results are fused with equal weighting, then re-ranked with ColBERT.

### Database Access Patterns

| Database | Operations |
|----------|------------|
| **MongoDB** | Document storage, BM25 text retrieval, node relationships, config checks |
| **Milvus** | Vector storage, semantic search, metadata filtering, collection management |
| **Fusion** | `QueryFusionRetriever` combines both via reciprocal rank fusion |

The architecture maintains separation of concerns: Milvus handles vector operations, MongoDB manages document text/metadata, with fusion bridging both systems.
