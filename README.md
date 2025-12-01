# RAG using LlamaIndex

A hybrid RAG system using LlamaIndex with sub-question decomposition, multi-tool query routing, and flexible metadata extraction for PDF document Q&A.

## Quick Start

### Prerequisites

- Python 3.11.1+
- Milvus 2.x (`http://localhost:19530`)
- MongoDB (`mongodb://localhost:27017/`)
- OpenAI and Anthropic API keys

### Installation

```bash
git clone https://github.com/tonypeng1/llamaindex.git
cd llamaindex
pip install uv && uv pip install -e .
```

### Setup

1. **Document**: Download [Paul Graham's essay](https://drive.google.com/file/d/1YzCscCmQXn2IcGS-omcAc8TBuFrpiN4-/view?usp=sharing) → `./data/paul_graham/paul_graham_essay.pdf`

2. **API Keys**: Create `.env`:
   ```
   OPENAI_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   ```

3. **Run**:
   ```bash
   python langextract_simple.py
   ```

---

## Features

| Feature | Description |
|---------|-------------|
| **Dual Storage** | Milvus (vectors) + MongoDB (documents) with deduplication |
| **Multi-Tool Query** | `keyphrase_tool` (facts), `page_filter_tool` (pages), `summary_tool` (summaries) |
| **Hybrid Retrieval** | Vector similarity + BM25 keyword search with reciprocal rank fusion |
| **Entity Filtering** | Auto-detects entities in queries, filters vector retriever (40-60% precision boost) |
| **Neural Re-ranking** | ColBERT for fine-grained relevance scoring |
| **KeyBERT** | Keyphrase extraction reduces BM25 noise |

---

## Configuration

### Metadata Extraction Options

| Option | Speed | Cost | Use Case |
|--------|-------|------|----------|
| `None` | ⚡⚡⚡ | Free | Quick testing |
| `"entity"` | ⚡⚡ | Free | Named entity recognition (local HuggingFace model) |
| `"langextract"` | ⚡ | API | Rich semantic metadata (concepts, advice, experiences) |
| `"both"` | ⚡ | API | Maximum metadata richness |

### Metadata Settings Example

```python
# langextract_simple.py
metadata = "langextract"           # None, "entity", "langextract", "both"
schema_name = "paul_graham_detailed"
use_entity_filtering = True
```

### RAG Pipeline Settings Example

```python
chunk_size = 256                   # Chunk size for splitting
chunk_overlap = 64                 # Overlap between chunks
similarity_top_k_fusion = 48       # Initial retrieval count
fusion_top_n = 42                  # Post-fusion count
rerank_top_n = 32                  # Final count after ColBERT
num_queries = 1                    # Query fan-out (1 = disabled)
num_nodes = 0                      # Neighbor nodes for context
```

---

## LangExtract Schema System

Schema definitions specify allowed metadata values. Managed in `langextract_schemas.py`.

### Operating Modes

| Mode | When | Source | Purpose |
|------|------|--------|---------|
| **Static** | Ingestion | Hardcoded | Guides extraction LLM |
| **Dynamic** | Query time | MongoDB | Ensures filters match stored values |

### Schema Attributes

| Attribute | Examples |
|-----------|----------|
| `concept_categories` | technology, startups, programming |
| `advice_domains` | career, creativity, relationships |
| `experience_periods` | childhood, college, viaweb, yc |
| `experience_sentiments` | positive, negative, mixed |
| `entity_roles` | founder, colleague, investor |
| `time_decades` | 1970s, 1980s, 1990s, 2000s |

### Key Functions

```python
# langextract_schemas.py
get_paul_graham_schema_definitions(use_dynamic_loading=True)
get_schema(schema_name)  # "paul_graham_detailed" or "paul_graham_simple"

# langextract_integration.py
extract_query_metadata_filters(query_str, schema_name)
```

---

## Architecture

### Hybrid Retrieval Pipeline

```
Query → [BM25 (MongoDB)] ─┬─→ Reciprocal Rank Fusion → ColBERT Re-rank → Response
        [Vector (Milvus)] ─┘
```

### Entity Filtering Behavior

| Retriever | Database | Entity Filter | Rationale |
|-----------|----------|---------------|-----------|
| BM25 | MongoDB | ❌ | Preserve keyword matches |
| Vector | Milvus | ✅ | Improve entity precision |

### Dynamic Filter Query Engine

When entity filtering is enabled, `DynamicFilterQueryEngine` wraps the retrieval pipeline to extract fresh metadata filters for **each sub-question** rather than using filters from the original query.

**Why it matters**: For multi-part questions, the SubQuestionQueryEngine decomposes this into separate sub-questions. Each sub-question gets its own entity/time filters, ensuring accurate retrieval for each time period.

**Flow** (example query: *"What did Paul Graham do in 1980, in 1996 and in 2019?"*):
```
Original Query → SubQuestionQueryEngine → Sub-question 1 ("What did Paul Graham do in 1980?")
                                        → Sub-question 2 ("What did Paul Graham do in 1996?")
                                        → Sub-question 3 ("What did Paul Graham do in 2019?")
                                              ↓
                           DynamicFilterQueryEngine extracts filters per sub-question
                                              ↓
                           Vector retriever uses sub-question-specific filters
```

**Key behavior**:
- Filters are extracted dynamically at query time (not pre-computed)
- Each sub-question gets independent entity recognition and filter creation
- BM25 retriever remains unfiltered; only vector retriever applies filters

### Database Responsibilities

| Database | Role |
|----------|------|
| **MongoDB** | Document storage, BM25 retrieval, node relationships |
| **Milvus** | Vector storage, semantic search, metadata filtering |

---

## File Structure

| Category | Files |
|----------|-------|
| **Core** | `langextract_simple.py`, `langextract_integration.py`, `langextract_schemas.py`, `utils.py`, `db_operation.py` |
| **Docs** | `README_GUIDE.md`, `EXAMPLES_METADATA.py` |
| **Tests** | `test/test_entity_filtering.py`, `test/test_langextract_install.py`, `test/test_langextract_schema.py`, `test/test_mongo_entity_metadata.py`, `test/demo_metadata_comparison.py`, `test/check_node_in_milvus.py`, `test/check_node_in_mongo.py`, `test/get_inclusive_schema.py` |

---

## Resources

- [Medium article](https://medium.com/@tony3t3t/rag-with-sub-question-and-tool-selecting-query-engines-using-llamaindex-05349cb4120c) with examples
- `README_GUIDE.md` for detailed documentation
- `EXAMPLES_METADATA.py` for code snippets
