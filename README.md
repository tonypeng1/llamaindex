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

#### MinerU Setup (Optional)
To use the MinerU parsing pipeline, create the isolated environment to avoid dependency conflicts:
```bash
uv venv .mineru_env
uv pip install -r requirements_mineru.txt --python ./.mineru_env/bin/python
```

### Setup

1. **Document**: Download [Paul Graham's essay](https://drive.google.com/file/d/1YzCscCmQXn2IcGS-omcAc8TBuFrpiN4-/view?usp=sharing) → `./data/paul_graham/paul_graham_essay.pdf`

2. **API Keys**: Create `.env`:
   ```
   OPENAI_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   ```

3. **Configure**: Edit `config.py` to select article and `queries.py` to select the active query:
   ```python
   # In config.py
   ACTIVE_ARTICLE = "paul_graham_essay"  # or other configured articles
   
   # In queries.py
   PG_ACTIVE = "What did the author do after handing off Y Combinator to Sam Altman?"
   ```

4. **Run**:
   ```bash
   # Index the document (MinerU)
   python minerU.py
   
   # Query the system
   python langextract_simple.py
   ```

---

## Features

| Feature | Description |
|---------|-------------|
| **Dual Storage** | Milvus (vectors) + MongoDB (documents) with deduplication |
| **Multi-Tool Query** | `keyphrase_tool` (facts), `page_filter_tool` (pages), `summary_tool` (summaries) |
| **Dynamic Filtering** | Extracts semantic filters (LangExtract) and entities per sub-question for precise retrieval |
| **Hybrid Retrieval** | Vector similarity + BM25 keyword search with reciprocal rank fusion |
| **Neural Re-ranking** | ColBERT for fine-grained relevance scoring |
| **KeyBERT** | Keyphrase extraction reduces BM25 noise |
| **MinerU Support** | Isolated PDF parsing pipeline optimized for Apple Silicon |

---

## Architecture & Data Processing

### The "Search-then-Fetch" Workflow
This system uses a dual-database architecture to balance search speed with context richness:

1.  **Milvus (The "Librarian"):** Stores mathematical **embeddings** and **Node IDs**. When you ask a question, Milvus identifies the most relevant IDs. It does *not* store or send the full text to the LLM.
2.  **MongoDB (The "Bookshelf"):** Stores the **full text** and **node relationships** (prev/next). Once Milvus finds the relevant IDs, the system "checks out" the full text from MongoDB.
3.  **Context Expansion:** If enabled, the system automatically retrieves neighboring nodes from MongoDB to provide the LLM with surrounding context.
4.  **Summary Path:** A separate MongoDB collection stores summary nodes, used exclusively by the `summary_tool` for "big picture" questions, bypassing the vector search entirely.

### Multimodal Handling (MinerU)
The pipeline explicitly handles complex document elements to ensure high-fidelity retrieval:

*   **Tables:** Extracted as HTML and converted to **Markdown**. This allows the LLM to reason about structured data naturally.
*   **Figures:** Uses a dual-path approach. The caption is included in the page text, while a **Claude Vision** agent generates a detailed visual description (labels, trends, diagrams) stored as a separate node.
*   **Equations:** Explicitly captured as **LaTeX** strings. This preserves mathematical precision for technical queries.

---

## Performance & Robustness

- **Split-Brain Protection** — Automatic detection and recovery from inconsistent database states (e.g., missing Milvus collection but existing MongoDB docstore) to ensure Node ID synchronization across all stores.
- **Metadata stripping** — strips large parser metadata (e.g., MinerU or LlamaParse OCR data) before saving to Milvus to avoid dynamic-field size and token-serialization blowups; full metadata is retained in MongoDB docstores.
- **Optimized node splitting** — uses a 512-token chunk size (aligned with ColBERT's limit) to ensure the reranker sees the entire context of every retrieved node.
- **Image processing & caching** — resizes large images, chooses appropriate format, base64-encodes payloads, and caches generated descriptions to avoid repeated vision API calls.
- **Safer embedding & lazy init** — reduced embedding batch sizes and deferred query-engine initialization to lower failure rates and startup cost.
- **Robust Sub-Question Generation** — uses `LLMQuestionGenerator` with a 3-attempt retry mechanism to handle transient JSON parsing failures during query decomposition.
- **Optional detailed responses** — set `RESPONSE_MODE=TREE_SUMMARIZE` for verbose structured answers (higher token use).

---

## Configuration

### Metadata Extraction Options

| Option | Speed | Cost | Use Case |
|--------|-------|------|----------|
| `None` | ⚡⚡⚡ | Free | Quick testing |
| `"entity"` | ⚡⚡ | Free | Named entity recognition (local HuggingFace model) |
| `"langextract"` | ⚡ | API | Rich semantic metadata (concepts, advice, experiences) |
| `"both"` | ⚡ | API | Maximum metadata richness |

### Centralized Configuration (`config.py`)

`config.py` is the **single source of truth** for the entire pipeline. Both the **Indexer** (`minerU.py`) and the **Retriever** (`langextract_simple.py`) import their settings from here to ensure database consistency.

#### 1. Select Active Article
Switch the entire system to a different document by changing one variable:

```python
# config.py
ACTIVE_ARTICLE = "paul_graham_essay"  # Options: paul_graham_essay, attention_all, etc.
```

#### 2. Global RAG Settings
Default parameters optimized for ColBERT reranking and multimodal retrieval:

```python
# config.py
DEFAULT_RAG_SETTINGS = {
    "chunk_size": 512,         # Aligned with ColBERT's 512-token limit
    "chunk_overlap": 128,
    "chunk_method": "sentence_splitter",
    "metadata": "None",        # Options: "None", "entity", "langextract", "both"
    "use_entity_filtering": False,
    "similarity_top_k_fusion": 35,
    "rerank_top_n": 15,
}
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
| **Core** | `langextract_simple.py`, `config.py`, `langextract_integration.py`, `langextract_schemas.py`, `utils.py`, `db_operation.py` |
| **Docs** | `README_GUIDE.md`, `EXAMPLES_METADATA.py` |
| **Tests** | `test/test_entity_filtering.py`, `test/test_langextract_install.py`, `test/test_langextract_schema.py`, `test/test_mongo_entity_metadata.py`, `test/demo_metadata_comparison.py`, `test/check_node_in_milvus.py`, `test/check_node_in_mongo.py`, `test/get_inclusive_schema.py` |

---

## Resources

- [Medium article](https://medium.com/@tony3t3t/rag-with-sub-question-and-tool-selecting-query-engines-using-llamaindex-05349cb4120c) with examples
- `README_GUIDE.md` for detailed documentation
- `EXAMPLES_METADATA.py` for code snippets
