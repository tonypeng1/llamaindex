# Changelog

## [December 22, 2025]

### 1. Migration to MinerU as Primary Parser

- **Main Script Promotion**: Promoted `minerU.py` to the primary indexing and RAG entry point, replacing `llamaparse.py`.
- **Code Archival**: Moved the original `llamaparse.py` to [archive/llamaparse.py](archive/llamaparse.py) to maintain a clean root directory while preserving the legacy pipeline.
- **Metadata Stripping**: Integrated `MetadataStripperPostprocessor` in `utils.py` to strip large parser metadata (OCR/Layout data) before Milvus insertion, preventing token serialization blowups.
- **Documentation Update**: Updated [README.md](README.md) and internal documentation to reflect the shift to MinerU as the default parsing engine.
- **Pipeline Simplification**: Removed dual-parser logic from the main script to focus on the high-performance MinerU/MLX pipeline, improving maintainability.

## [December 21, 2025]

### 1. MinerU Integration for PDF Parsing

**Isolated Environment Strategy**:
- Implemented an **isolation strategy** for MinerU (Magic-PDF) to resolve deep dependency conflicts with the main environment (Torch/Accelerate).
- Created a dedicated `.mineru_env` virtual environment and a `mineru_wrapper.py` script to execute MinerU CLI via `subprocess`.
- Optimized for **Apple Silicon** using the `vlm-mlx-engine` backend for high-performance parsing.

**MinerU Pipeline Integration**:
- Added `load_document_mineru` to convert MinerU's structured JSON output into LlamaIndex `Document` objects, maintaining page-level grouping.
- Implemented `load_image_text_nodes_mineru` to extract images from MinerU output and generate detailed descriptions using **Claude Vision API** (`claude-sonnet-4-20250514`), ensuring parity with the LlamaParse pipeline.
- Developed `build_section_index_mineru` to support **deterministic section retrieval** for MinerU-parsed documents, enabling exact section-based queries.

**Pipeline Flexibility**:
- Introduced a `chunk_method` toggle in `llamaparse.py` to switch between `"llamaparse"` and `"mineru"` pipelines.
- Standardized metadata and node structures across both parsers to allow seamless comparison of RAG performance.

### 2. RAG Optimization & Prompt Engineering

- **Centralized Configuration**: Moved all RAG, database, and embedding settings to `config.py` as a single source of truth.
- **Chunking Strategy**: Standardized on a **512-token chunk size** (aligned with ColBERT) and **80-token overlap** to improve retrieval context.
- **LaTeX Standardization**: Updated all response prompts to enforce `$$ ... $$` for display equations and `$ ... $` for inline math.
- **Dynamic Indexing**: Updated collection naming to include chunk parameters, preventing index collisions when settings are tuned.
- **Improved Retrieval**: Refined deterministic section matching by sorting keys by length and replaced `GuidanceQuestionGenerator` with `LLMQuestionGenerator` for more reliable sub-query decomposition.

### 3. Robustness & Fixes

- **Async Support**: Fixed `LazyQueryEngine` to properly handle async queries and initialization.
- **Metadata Handling**: Enhanced `PageSortNodePostprocessor` to handle non-integer page numbers gracefully.
- **Table Processing**: Added HTML-to-Markdown conversion for MinerU tables using `pandas` for better LLM reasoning.

## [December 20, 2025]

### 1. Enhanced Section Indexing and Retrieval

**Improved Heading Detection**: Significantly upgraded `build_section_index` in `llamaparse.py` to handle diverse document formats:
- Added support for **Markdown** (`#`), **plain numbered** (`5 CONCLUSION`), and **LaTeX** (`\section{...}`) headings.
- Implemented **uppercase fallback** for titles like "ACKNOWLEDGEMENTS" that lack numbering.
- Expanded search to multiple fields (`md`, `text`, `value`) and page-level sources to ensure no sections are missed.

**Deterministic Section Retrieval**:
- Added a **pre-check logic** in `page_filter_tool` that resolves section-based queries (e.g., "Summarize the CASE STUDIES section") directly from the `section_index`.
- This bypasses the LLM for known sections, improving **speed, reliability, and cost** while avoiding non-deterministic LLM failures.
- Maintained LLM fallback for complex page-range queries (e.g., "pages 5-10").

---

## Changes Since Last Commit (December 14, 2025)

### 1. Fixed Critical Token Overflow Issue

**Problem**: Queries were failing with `prompt is too long: 215512 tokens > 200000 maximum` error despite node content being only ~7K tokens.

**Root Cause**: LlamaParse nodes contained massive hidden metadata (OCR coordinates, bounding boxes, table structures) that got serialized when sent to the LLM, causing a **36x token blowup** (6,920 content tokens → 253,372 serialized tokens).

**Solution**: Added `MetadataStripperPostprocessor` in `utils.py` that creates clean `TextNode` objects with only text content and minimal metadata, reducing serialized tokens by **96%** (253K → 11K).

#### Token Analysis (25 nodes, original parameters)

| Metric | Before Fix | After Fix | Improvement |
|--------|------------|-----------|-------------|
| Content Tokens | 6,920 | 6,920 | Same |
| Serialized Tokens | 253,372 | 10,787 | **96% reduction** |
| Worst Node (Page 13) | 75,944 | 1,698 | **98% reduction** |

---

### 2. New Classes in `utils.py`

#### `MetadataStripperPostprocessor`
A postprocessor that strips large metadata from nodes before LLM synthesis:
- Extracts only essential metadata (page number, type)
- Creates clean `TextNode` objects to avoid serialization bloat
- Fixes LlamaParse nodes that contain huge OCR coordinate arrays

#### `PrintNodesPostprocessor` (Enhanced)
Updated to show both content tokens and serialized tokens for debugging:
- Shows token breakdown sorted by size
- Displays content vs serialized token comparison
- Warns when approaching Claude's 200K token limit

---

### 3. Major Rewrite of `llamaparse.py`

The main script was significantly restructured from a simple 2-tool architecture to a more sophisticated 3-tool RAG system.

#### New Features

- **3-Tool Architecture**: Replaced simple `recursive_query_tool` + `summary_tool` with:
  - `keyphrase_tool` - BM25 + Vector fusion for specific queries (equations, figures, tables)
  - `summary_tool` - Full document context summarization
  - `page_filter_tool` - Section/page-specific retrieval

- **LazyQueryEngine Class** - Defers tool initialization until first query to prevent eager initialization

- **Section Index System** - New `build_section_index()` and `format_section_list_for_prompt()` functions that parse LlamaParse JSON to map section names to page ranges

- **Custom Guidance Prompt** - `create_custom_guidance_prompt()` for SubQuestionQueryEngine with page-based examples

- **Image Processing with Caching** - `load_image_text_nodes_llamaparse()` extracts images and generates detailed descriptions via Anthropic's vision API (raw `anthropic` SDK to avoid package conflicts). Descriptions are cached to `{article_name}_image_descriptions.json` (in the article data folder) to avoid repeated expensive API calls. Images exceeding Claude's dimension limits are resized (max dimension ~8000 px), saved in an appropriate format (PNG/JPEG), and base64-encoded before sending, reducing payload size and improving reliability; trade-off: initial cache generation cost and possible minor fidelity loss from resizing.

- **Node Size Management** - Added `SentenceSplitter` to handle oversized nodes that exceed embedding model limits (8192 tokens). Uses a conservative splitting threshold (~8000 characters ≈ 2000 tokens) with overlapping chunks (chunk_size=2000, chunk_overlap=100) to avoid embedding failures; trade-off: more chunks → more embeddings and retrieval candidates.

- **Metadata stripping before vector writes** - Large metadata (OCR coordinates, bounding boxes, table structures) is stripped from nodes before saving to Milvus to avoid exceeding Milvus dynamic-field size and serialization/token blowups; full metadata remains available in MongoDB docstores.

- **Prev/Next stitching** - `stitch_prev_next_relationships()` is applied before saving nodes to both vector and summary MongoDB docstores so the PrevNextNodePostprocessor can retrieve neighboring nodes (enabled when `add_document_vector`/`add_document_summary` are true).

- **Response synthesis mode** - Added `RESPONSE_MODE` environment variable to select the response synthesizer. When set to `TREE_SUMMARIZE`, a more detailed summary prompt template is used to produce verbose, structured answers (improves answer quality but increases token usage and latency).

- **LLM Model Update** - Changed from `claude-3-5-sonnet-20240620` to `claude-sonnet-4-0`

- **Embedding Batch Size** - Reduced to 10 to avoid 300k token limit

- **Premium Mode Parsing** - Added detailed parsing instructions for complex math symbols, LaTeX handling

- **Retry Logic** - Added retry mechanism for GuidanceQuestionGenerator JSON parsing failures

- **Active Article** - Changed from "attention" paper to "RAG_Anything" paper

---

### 2. Updates to `utils.py`

- **PageSortNodePostprocessor** - Now supports both `'page'` (LlamaParse) and `'source'` (PyMuPDF) metadata keys
- Added `stitch_prev_next_relationships()` helper and use it to stitch prev/next relationships before saving nodes to MongoDB docstores (improves neighbor retrieval in post-processing).
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
num_nodes_prev_next = 1       # Neighboring nodes (1 = enabled by default)
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
- Extracts images and generates detailed descriptions using Claude Vision; descriptions cached to `{article_name}_image_descriptions.json` (stored in the article data folder) to prevent repeated API calls and lower runtime cost
- Resizes images that exceed Claude's max pixel dimension (~8000 px), selects an appropriate format (PNG/JPEG), and base64-encodes payloads before sending to the API—this reduces payload size and API errors; caching + resizing lower costs and improve reliability but may slightly reduce fidelity for very large images
- Graceful error handling and cache lookups are used to skip API calls when descriptions exist
