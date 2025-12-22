# LlamaIndex RAG Implementation Guide

## Table of Contents
- [Quick Start](#quick-start)
    - [Requirements](#requirements)
    - [Setup & Run](#setup--run)
    - [Configuration](#configuration)
    - [Decision Tree](#decision-tree)
- [Metadata Extraction Options](#metadata-extraction-options)
    - [Option 1: None (Basic)](#option-1-none-basic)
    - [Option 2: EntityExtractor](#option-2-entityextractor)
    - [Option 3: LangExtract](#option-3-langextract)
    - [Option 4: Both](#option-4-both-entityextractor--langextract)
- [Entity-Based Filtering](#entity-based-filtering)
    - [Overview](#overview)
    - [Architecture](#architecture)
    - [How It Works](#how-it-works)
    - [Configuration](#configuration-1)
    - [Advanced Configuration](#advanced-configuration)
    - [Supported Entities](#supported-entities)
    - [Query Examples](#query-examples)
    - [Processing Pipeline](#processing-pipeline)
- [Testing](#testing)
    - [Test Entity Filtering](#test-entity-filtering)
    - [Test Workflow](#test-workflow)
    - [Query Examples by Type](#query-examples-by-type)
- [Files in This Implementation](#files-in-this-implementation)

---

## Quick Start

### Requirements

- Python 3.11.1+ (see `pyproject.toml`)
- Access to Milvus 2.x at `http://localhost:19530`
- Access to MongoDB at `mongodb://localhost:27017/`
- Paul Graham PDF placed at `./data/paul_graham/paul_graham_essay.pdf`
- `.env` containing `OPENAI_API_KEY` (LangExtract or both) and `ANTHROPIC_API_KEY`

### Setup & Run

1. Install dependencies (`uv pip install -e .` or `pip install -e .`).
2. Start Milvus and MongoDB (update URIs in `config.py` → `DATABASE_CONFIG` if your endpoints differ).
3. Download the article and place it under `data/paul_graham/` as shown above.
4. Create/update `.env` with required API keys.
5. (Optional) Select article in `config.py`:
    ```python
    ACTIVE_ARTICLE = "paul_graham_essay"  # Change to switch articles
    ```
6. Run the pipeline:
    ```bash
    python langextract_simple.py
    ```

### Configuration

All configuration is centralized in `config.py`:

```python
# Select active article
ACTIVE_ARTICLE = "paul_graham_essay"

# Available articles (pre-configured):
# - paul_graham_essay, How_to_do_great_work (Paul Graham essays)
# - attention_all, metagpt, uber_10q_march_2022, eBook-How-to-Build-a-Career-in-AI

# Default RAG settings (applied to all articles)
DEFAULT_RAG_SETTINGS = {
    "metadata": "None",             # None, "entity", "langextract", "both"
    "use_entity_filtering": False,
    "chunk_size": 512,
    "chunk_overlap": 128,
    "similarity_top_k_fusion": 35,
    "rerank_top_n": 15,
}

# Per-article overrides (optional)
ARTICLE_RAG_OVERRIDES = {
    "attention_all": {"metadata": "entity", "chunk_size": 512},
}

# Database endpoints
DATABASE_CONFIG = {
    "milvus_uri": "http://localhost:19530",
    "mongo_uri": "mongodb://localhost:27017/",
}
```

**Helper functions:**
```python
from config import get_active_config, get_rag_settings, print_article_summary
```

### Decision Tree

```
Do you need metadata?
    ├─ NO  → metadata = None (Fast, FREE)
    └─ YES → Do you have budget for GPT-4 API?
         ├─ NO  → metadata = "entity" (Fast, FREE)
         └─ YES → Need semantic metadata?
              ├─ NO  → metadata = "entity"
              └─ YES → Need both entities & semantic?
                   ├─ NO  → metadata = "langextract"
                   └─ YES → metadata = "both"
```

---

## Metadata Extraction Options

### Option 1: None (Basic)

**When to use:** Quick testing, simple documents

```python
metadata = None
```

**Output:**
- Basic chunking with page numbers only
- Processing time: ~10 seconds for 30 pages
- Cost: **FREE**

**Metadata example:**
```python
{
    'source': '1',
    'file_path': '/path/to/document.pdf',
    'file_name': 'document.pdf',
    'page_label': '1'
}
```

---

### Option 2: EntityExtractor

**When to use:** Need entity recognition, want free solution

```python
metadata = "entity"
```

**Output:**
- Named entities: PERSON, ORGANIZATION, LOCATION
- Uses local HuggingFace model
- Cost: LLM API calls

**Metadata example:**
```python
{
    'source': '1',
    'file_path': '/path/to/document.pdf',
    'PER': ['Paul Graham', 'Jessica Livingston'],
    'ORG': ['Y Combinator', 'Viaweb'],
    'LOC': ['Silicon Valley', 'Cambridge']
}
```

---

### Option 3: LangExtract

**When to use:** Need deep semantic understanding, complex queries

```python
metadata = "langextract"
schema_name = "paul_graham_detailed"  # or "paul_graham_simple"
```

**Output:**
- Concepts, advice, experiences, entities, time references
- Cost: LLM API calls

**Metadata example:**
```python
{
    'source': '1',
    'langextract_concepts': ['startup ecosystem', 'programming'],
    'concept_categories': ['technology', 'business'],
    'langextract_advice': ['focus on product-market fit'],
    'advice_types': ['strategic'],
    'langextract_entities': ['Y Combinator'],
    'entity_roles': ['organization'],
    'time_references': ['1995', '2000s']
}
```

**Available schemas:**
- `paul_graham_detailed` - Rich comprehensive metadata (slower, more expensive)
- `paul_graham_simple` - Basic semantic metadata (faster, cheaper)

---

### Option 4: Both (EntityExtractor + LangExtract)

**When to use:** Maximum metadata richness

```python
metadata = "both"
schema_name = "paul_graham_detailed"
```

**Output:**
- All EntityExtractor + All LangExtract metadata
- Cost: LLM API calls

---

## Entity-Based Filtering

### Overview

Entity filtering enhances retrieval precision by filtering the **vector retriever** to only nodes mentioning specific entities (people, organizations, locations) from the user's query. The system uses a **hybrid approach**: BM25 retriever operates on the full docstore (no filtering) while the vector retriever can be entity-filtered, then both results are fused together.

### Architecture

The system uses **two parallel retrievers** that are always fused together:

1. **BM25 Retriever (Keyword-based)**
   - Operates on: MongoDB docstore
   - Filtering: NO entity filtering (always searches full docstore)
   - Strategy: Keyphrase extraction → keyword matching
   - Purpose: Ensure keyword-relevant results aren't missed

2. **Vector Retriever (Semantic)**
   - Operates on: Milvus vector index
   - Filtering: Optional entity filtering (when `use_entity_filtering = True`)
   - Strategy: Embedding similarity → semantic matching
   - Purpose: Find semantically relevant results, optionally filtered by entities

3. **Fusion Layer**
    - Uses `QueryFusionRetriever` in `reciprocal_rerank` mode to merge rankings
    - Automatically balances BM25 + vector scores (no manual weighting)
    - Followed by ColBERT reranking for final precision

### How It Works

```
User Query: "What did Paul Graham advise about Y Combinator?"
    ↓
Extract entities: ["Paul Graham" (PER), "Y Combinator" (ORG)]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: BM25 Retrieval (Keyword-based)                     │
│   • Keyphrase extraction (KeyBERT) for long query          │
│   • Operates on MongoDB docstore                           │
│   • NO entity filtering                                    │
│   • Retrieves nodes matching keyphrases or original query  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Vector Retrieval (Semantic)                        │
│   • Operates on Milvus vector index                        │
│   • Optional entity filters (only nodes with entities)     │
│   • Retrieves semantically similar nodes                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Fusion                                              │
│   • QueryFusionRetriever (reciprocal-rank fusion)          │
│   • BM25 results (unfiltered) + Vector results (filtered)  │
│   • Automatic score balancing (no manual weights)          │
└─────────────────────────────────────────────────────────────┘
    ↓
ColBERT reranking
    ↓
High-precision results! 🎯
```

**Key Points:**
- **BM25**: Always operates on full MongoDB docstore, no entity filtering
- **Vector**: Can be entity-filtered when `use_entity_filtering = True`
- **Fusion**: `QueryFusionRetriever` (reciprocal rank) merges both approaches every time
- **Result**: Balance between keyword matching (BM25) and semantic relevance (Vector)

### Configuration

```python
# Enable entity metadata extraction
metadata = "entity"  # or "langextract" or "both"

# Enable entity filtering
use_entity_filtering = True
```

### Advanced Configuration

You can fine-tune the retrieval pipeline with these parameters in `langextract_simple.py`:

```python
# 1. Retrieval Parameters
similarity_top_k_fusion = 35  # Initial retrieval count from vector store
fusion_top_n = 25             # Number of nodes to keep after fusing BM25 + Vector results
rerank_top_n = 15             # Final number of nodes after ColBERT reranking
num_queries = 1               # Fusion query fan-out (1 disables query generation)
num_nodes = 0                 # Additional context via SafePrevNext postprocessor

# 2. Chunking Strategy
chunk_size = 512              # Standardized on 512 for ColBERT alignment
chunk_overlap = 128           # Overlap to maintain context across chunks
```

### Supported Entities

**People (PER):**
- Paul Graham, Jessica Livingston, Robert Morris, Trevor Blackwell, Sam Altman

**Organizations (ORG):**
- Y Combinator, YC, Viaweb, Yahoo, MIT, Harvard, RISD, Interleaf

**Locations (LOC):**
- Silicon Valley, Cambridge, San Francisco, New York, Florence (Italy)

**Extensible:** Add new entities in `utils.py` → `extract_entities_from_query()`

### Query Examples

✅ **Highly Effective:**
```python
"What did Paul Graham advise about Y Combinator?"
"Where did Jessica Livingston work before YC?"
"What happened at Viaweb and Yahoo?"
```

⚠️ **Less Effective (No Entities):**
```python
"What advice is given about startups?"
"Summarize the document"
```

### Processing Pipeline

```
OPTION 1: None
Load PDF → Split Chunks → Basic Metadata → Store to DB

OPTION 2: EntityExtractor
Load PDF → Split Chunks → Entity Extractor → Entity Metadata → Store to DB

OPTION 3: LangExtract
Load PDF → Split Chunks → LangExtract (GPT-4) → Semantic Metadata → Store to DB

OPTION 4: Both
Load PDF → Split Chunks → Entity Extractor → LangExtract → Combined Metadata → Store to DB
```

## Testing

### Test Entity Filtering
```bash
python test_entity_filtering.py
```

### Test Workflow
```python
# Step 1: Start with None or EntityExtractor
metadata = None  # or "entity"
use_entity_filtering = False # or True

# Step 2: Test LangExtract on small sample (2-3 pages)
metadata = "langextract"
schema_name = "paul_graham_simple"
use_entity_filtering = True

# Step 3: Scale up to full document
metadata = "langextract"
schema_name = "paul_graham_detailed"
use_entity_filtering = True

# Step 4: Choose for production based on requirements
```

### Query Examples by Type

**Basic Queries (None):**
```python
"What is on page 5?"
"Summarize the entire document"
```

**Entity-Based Queries (EntityExtractor):**
```python
"What companies are mentioned?"
"List all people mentioned on pages 10-15"
"Where did the author work?"
```

**Semantic Queries (LangExtract):**
```python
"What strategic advice is given about startups?"
"What experiences from the 1990s are described?"
"What programming concepts are discussed?"
```

**Complex Queries (Both):**
```python
"What did Paul Graham advise about startup culture?"
"How do the entities relate to the advice given in the 2000s?"
"What experiences led to strategic advice about product development?"
```

---

## Files in This Implementation

- **`langextract_simple.py`** - Main RAG implementation
- **`langextract_integration.py`** - LangExtract integration functions
- **`langextract_schemas.py`** - Extraction schemas
- **`utils.py`** - Utility functions including entity filtering
- **`test_entity_filtering.py`** - Entity filtering tests
- **`EXAMPLES_METADATA.py`** - Code examples and quick-start guide
- **`README_GUIDE.md`** - This comprehensive guide

