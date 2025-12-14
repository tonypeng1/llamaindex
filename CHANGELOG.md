# Changelog

## Changes Since Last Commit (December 14, 2025)

### 1. Major Rewrite of `llamaparse.py`

The main script was significantly restructured from a simple 2-tool architecture to a more sophisticated 3-tool RAG system.

#### New Features

- **3-Tool Architecture**: Replaced simple `recursive_query_tool` + `summary_tool` with:
  - `keyphrase_tool` - BM25 + Vector fusion for specific queries (equations, figures, tables)
  - `summary_tool` - Full document context summarization
  - `page_filter_tool` - Section/page-specific retrieval

- **LazyQueryEngine Class** - Defers tool initialization until first query to prevent eager initialization

- **Section Index System** - New `build_section_index()` and `format_section_list_for_prompt()` functions that parse LlamaParse JSON to map section names to page ranges

- **Custom Guidance Prompt** - `create_custom_guidance_prompt()` for SubQuestionQueryEngine with page-based examples

- **Image Processing with Caching** - `load_image_text_nodes_llamaparse()` now uses raw Anthropic SDK (avoiding package conflicts) with image description caching

- **Node Size Management** - Added `SentenceSplitter` to handle oversized nodes that exceed embedding model limits (8192 tokens)

- **LLM Model Update** - Changed from `claude-3-5-sonnet-20240620` to `claude-sonnet-4-0`

- **Embedding Batch Size** - Reduced to 10 to avoid 300k token limit

- **Premium Mode Parsing** - Added detailed parsing instructions for complex math symbols, LaTeX handling

- **Retry Logic** - Added retry mechanism for GuidanceQuestionGenerator JSON parsing failures

- **Active Article** - Changed from "attention" paper to "RAG_Anything" paper

---

### 2. Updates to `utils.py`

- **PageSortNodePostprocessor** - Now supports both `'page'` (LlamaParse) and `'source'` (PyMuPDF) metadata keys

- **New LlamaParse-specific Functions**:
  - `get_database_and_llamaparse_collection_name()`
  - `get_llamaparse_vector_store_docstore_and_storage_context()`
  - `change_default_engine_prompt_to_in_detail()`
  - `display_prompt_dict()`

---

### 3. Dependency Updates

- Added `anthropic>=0.72.0` to `pyproject.toml` and `uv.lock`

---

### 4. New Test Scripts

- `test/test_equation_parsing.py` - Tests different LlamaParse settings for equation recognition
- `test/test_node_sizes.py` - Analyzes node sizes from cached LlamaParse data

---

### 5. Archived Old Code

- Moved old `llamaparse.py` from archive folder to `archive/llamaparse_archived.py`
- Deleted `archive/llamaparse.py`

---

## Technical Details

### 3-Tool Architecture

| Tool | Purpose | Use Cases |
|------|---------|-----------|
| `keyphrase_tool` | BM25 + Vector hybrid retrieval | Equations, figures, tables, specific facts |
| `summary_tool` | Full document tree summarization | Document overview, main themes |
| `page_filter_tool` | Section/page-specific retrieval | Specific sections, page ranges |

### Key Configuration Parameters

```python
# Fusion retrieval parameters
similarity_top_k_fusion = 48  # Nodes for fusion
fusion_top_n = 42             # Nodes after fusion
num_queries_fusion = 1        # Query generation (1 = disabled)
rerank_top_n = 32             # Nodes after ColBERT reranking
num_nodes_prev_next = 0       # Neighboring nodes (0 = disabled)
```

### LlamaParse Configuration

```python
parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    parsing_instruction=parsing_instruction_equations,
    premium_mode=True,
    invalidate_cache=False,
    verbose=True,
)
```

### Image Processing

- Uses raw `anthropic` SDK instead of `AnthropicMultiModal` to avoid package conflicts
- Caches image descriptions to `{article_name}_image_descriptions.json`
- Handles image resizing for Claude's 8000px dimension limit
