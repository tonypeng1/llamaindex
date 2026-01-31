# Changelog

## [0.2.1] - 2026-01-31

### Vision Model Upgrade: Gemini 3 Flash Agentic Vision

- **New Vision API**: Migrated image description from Claude Sonnet (Anthropic) to **Gemini 3 Flash Agentic Vision** (Google) in [main.py](main.py). The new model uses Google's native `google-genai` SDK with code execution enabled, allowing the model to zoom, crop, and programmatically analyze fine image details.
- **Agentic Vision Support**: Enabled Gemini's Think-Act-Observe loop for image analysis, which generates and executes Python code server-side for detailed inspection of figures, diagrams, and charts.
- **Multi-Part Response Handling**: Added logic to parse Agentic Vision's multi-part responses (which include `executable_code`, `code_execution_result`, and `text` parts), extracting only the textual description.
- **Retry Logic with Exponential Backoff**: Implemented robust error handling for transient 503 UNAVAILABLE errors from the Gemini API (3 retries with 5s/10s/20s delays).
- **Progress Indicators**: Added emoji-based progress logging during image processing: `✅` for cached, `🔄` for processing, `⏳` for retries.
- **New Environment Variable**: Added `GOOGLE_API_KEY` to [.env.example](.env.example) and updated [README.md](README.md) prerequisites.

### Documentation

- **README Showcase Section**: Added a new "Showcase: Multimodal Understanding in Action" section to [README.md](README.md) demonstrating the pipeline's ability to describe complex technical diagrams using Gemini 3 Flash Agentic Vision.
- **RAG-Anything Example**: Included a detailed use case analyzing Figure 1 from the [RAG-Anything paper](https://arxiv.org/abs/2510.12323), showcasing the system's multimodal comprehension capabilities.
- **Framework Image**: Added `images/rag_anything_framework.png` and updated [.gitignore](.gitignore) to selectively track this image while ignoring other files in the `images/` folder.

### Dependencies

- Added `google-generativeai>=0.8.6` and `llama-index-multi-modal-llms-gemini>=0.6.1` to [pyproject.toml](pyproject.toml).
- Relaxed `pillow` constraint from `==12.0.0` to `>=10.2.0,<13.0.0` to resolve dependency conflicts.

## [0.2.0] - 2026-01-24

### Simplified Onboarding & Automated Infrastructure
- **Automated setup script**: Introduced [setup.sh](setup.sh) to handle one-click environment creation, including `.env` generation from a template and automatic creation of the isolated `.mineru_env` virtual environment.
- **One-click Demo**: Enhanced [setup.sh](setup.sh) to automatically download a demo PDF (Paul Graham essay) and create the required directory structure, making the project runnable immediately after setup.
- **Environment template**: Added [.env.example](.env.example) to provide a clear starting point for API keys and database configurations.
- **Safe onboarding logic**: Updated [main.py](main.py) to detect missing PDFs and provide direct download links, ensuring a smooth experience for first-time users.
- **Safe environment updates**: Configured [setup.sh](setup.sh) to detect and skip existing virtual environments (`.venv`, `.mineru_env`) to prevent accidental overwrites for active developers.

### Database & Docker Management
- **Unified database stack**: Updated [db.sh](db.sh) to include a managed **MongoDB** container, providing a complete local infrastructure (Milvus + MongoDB + Attu) in one command.
- **Homebrew compatibility**: Implemented automatic port detection for port `27017`. The script now detects and respects native Homebrew MongoDB installations, gracefully skipping the Docker version to avoid conflicts.
- **Zero-touch Docker**: Added macOS auto-launch logic. The script now automatically starts Docker Desktop and waits for the daemon to be ready if it finds it's not running.
- **Start/Stop All Commands**: Consolidated daily operations under `start_all` and `stop_all` arguments in [db.sh](db.sh).

## [January 18, 2026] - Robust Technical Retrieval & Multi-Reference Tooling

### RAG Retrieval Improvements

- **Preserved Decimals in Normalization**: Updated `normalize_for_matching` in [rag_factory.py](rag_factory.py) to keep decimal points between digits (e.g., "4.3"), ensuring exact matches for technical figure and equation labels.
- **Multi-Reference Page Detection**: Enhanced `page_filter_tool` to use `re.findall`, allowing it to identify and resolve multiple figure/equation references (e.g., "Figure 4.1 and 4.3") into a single set of page filters.
- **Regex Filter Fallback**: Added manual regex extraction for `figure_label` and `equation_label` in `DynamicFilterQueryEngine`. This provides a reliable fallback for metadata filtering when the NER model (GLiNER) fails to identify specific figure/equation entities in the query.
- **Improved Metadata Filtering**: Ensured that identified figure labels are consistently applied as metadata filters (`FilterOperator.EQ`) during vector retrieval, leading to more accurate context for specific technical queries.

### Bug Fixes

- Fixed an issue where "Figure 4.3" was omitted from retrieval because the decimal point was being stripped during the normalization process.
- Resolved a limitation in `page_filter_tool` where only the first figure reference in a query was being used to filter page context.

## [January 2, 2026] - Figure & Image Retrieval Improvements

### Highlights

- Image description nodes now include the original image caption in the node text and a normalized `figure_label` metadata field when a figure label is detected (e.g., `4.1`).
- Deterministic figure reference detection now supports decimal labels (e.g., `fig. 4.1`) and `_find_pages_for_reference` prefers `figure_label` metadata for reliable page resolution.
- Queries that reference figures will include matching image nodes in BM25 and vector retrieval (via `figure_label` filters and BM25 pool injection), reducing omissions during synthesis.
- Minor API/config updates: added `ACTIVE_ARTICLE` usage in `main.py`, and `get_database_and_llamaparse_collection_name()` now accepts `article_key` for deterministic collection names.
- Added `check_node_ids.py` debug helper to inspect vector/summary docstores for node presence.

## [January 1, 2026] - New Article Integration: Laser_coprop_RA

### 1. Configuration Updates

- Added support for new academic paper "Laser_coprop_RA" (Furukawa team's iGM laser for DRA) in [config.py](config.py) and [queries.py](queries.py).
- Configured the article with academic schema, both metadata extraction (GLiNER + LangExtract), and entity filtering enabled.
- Set optimized RAG settings: 256-token chunks, 64-token overlap, and 1 prev/next node for context.

### 2. Query Addition

- Added active query for "Laser_coprop_RA": "What is the signal wavelength and pump wavelength used in the experiments described in the paper? Infer from contents in figures if necessary."

### 3. File Requirement

- Requires adding `Laser_coprop_RA.pdf` to `/data/DRA/` folder for ingestion.

## [December 27, 2025] - RAG Synthesis Optimization & Schema Expansion

### 1. RAG Engine Enhancements

- **Node Sorting & Deduplication**: Introduced `SortedResponseSynthesizer` in [rag_factory.py](rag_factory.py). This wrapper deduplicates retrieved nodes by ID and sorts them by page number before synthesis, ensuring more coherent and chronologically accurate responses.
- **Sorted Sub-Question Engine**: Implemented `SortedSubQuestionQueryEngine` to automatically apply the sorting and deduplication logic to the final response synthesis of multi-step queries.
- **Diagnostic Logging**: Added better visibility into GLiNER entity extraction results and LangExtract filter matching during the query process.

### 2. Unified Extraction Schema System

- **GLiNER Integration**: Significantly expanded [extraction_schemas.py](extraction_schemas.py) to serve as a central hub for both LangExtract and GLiNER. Added `GLINER_ENTITY_SETS` with optimized labels for Academic, Technical, Financial, and Career domains.
- **Expanded Domain Support**: Added static definitions for `financial`, `career`, `technical`, and `general` document types to the unified schema registry.
- **Dynamic Loading**: Enhanced support for fetching distinct metadata values from MongoDB to keep extraction attributes synchronized with the database.

### 3. Configuration & Query Updates

- **Active Query Update**: Updated the default test query for Paul Graham essays in [queries.py](queries.py) to focus on location-based event extraction.
- **LangExtract Logging**: Added explicit logging in [langextract_integration.py](langextract_integration.py) to indicate when no metadata filters are found for a query.

## [December 24, 2025] - Schema Reorganization & File Rename

### 1. Major Refactoring

- **File Renamed**: `minerU.py` → `main.py` for clearer entry point semantics. All README and documentation updated accordingly.
- **Schema Module Expansion**: Renamed and significantly expanded `langextract_schemas.py` → `extraction_schemas.py` with:
  - Multi-domain support: Paul Graham essays, Academic papers, Technical docs, Financial reports, Career guides, and General documents
  - Domain-specific GLiNER entity sets (~20 types per domain) for optimized entity extraction
  - Unified schema registry with backward-compatible functions
  - Dynamic schema loading from MongoDB for query-time validation
- **Updated Imports**: Propagated `extraction_schemas` import across `rag_factory.py`, `test_gliner_mineru.py`, `test_gliner_query.py`, and `test_langextract_schema.py`.
- **Query Configuration**: Updated default test query in `queries.py` to focus on actionable career advice.

### 2. Documentation Updates

- Updated README with `python main.py` instead of `python minerU.py` (3 locations)

## [December 24, 2025]

### 1. GLiNER Entity Extraction Integration

- **GLiNER Transition**: Replaced the `span-marker` based `EntityExtractor` with a custom `GLiNERExtractor` in [gliner_extractor.py](gliner_extractor.py). This enables zero-shot, domain-specific entity extraction using the `urchade/gliner_medium-v2.1` model on Apple Silicon (MPS).
- **Domain-Specific Entity Sets**: Defined six specialized entity sets (Academic, Technical, Financial, General, Paul Graham, and Career) in [extraction_schemas.py](extraction_schemas.py), each containing ~20 optimized entity types for better retrieval precision.
- **MinerU Ingestion Integration**: Integrated GLiNER into the [minerU.py](minerU.py) pipeline. Both base text nodes and image description nodes are now enriched with domain-specific entities during ingestion.
- **Dynamic Query Filtering**: Updated [rag_factory.py](rag_factory.py) to use GLiNER for query-time entity extraction. This allows the `DynamicFilterQueryEngine` to apply precise metadata filters to sub-questions based on the detected entities.

### 2. Configuration & Testing

- **Active Article Switch**: Updated [config.py](config.py) to set `paul_graham_essay` as the default `ACTIVE_ARTICLE` and enabled `entity` metadata extraction by default.
- **New Test Suite**: Added [test/test_gliner_mineru.py](test/test_gliner_mineru.py) and [test_gliner_query.py](test_gliner_query.py) to validate GLiNER extraction quality and performance across different document types.
- **Dependency Update**: Added `gliner` to [pyproject.toml](pyproject.toml) and updated the lockfile.

## [December 23, 2025]

### 1. LangExtract Caching & MinerU Integration

- **EntityExtractor Integration**: Added full support for local, model-based entity extraction in the MinerU pipeline. Both base text nodes and image description nodes are now processed using the `lxyuan/span-marker-bert-base-multilingual-cased-multinerd` model on Apple Silicon (MPS).
- **Metadata Caching**: Implemented `enrich_nodes_with_langextract_cached` in `langextract_integration.py`. This uses a local JSON cache (`_langextract_cache.json`) to store GPT-4o extraction results, significantly reducing API costs and ingestion time for repeated runs.
- **MinerU LangExtract Support**: Integrated LangExtract enrichment into the `minerU.py` pipeline. Both base text nodes and image description nodes can now be enriched with structured semantic metadata.
- **Dynamic Collection Naming**: Updated `utils.py` and `minerU.py` to include the `metadata` extraction method in the database collection names. This prevents data collisions when switching between different metadata extraction strategies (e.g., None vs. LangExtract).
- **Logging Optimization**: Suppressed verbose `absl` (Google) logging across `minerU.py` and `langextract_simple.py` for a cleaner terminal experience.

### 2. Configuration & Query Refinement

- **Active Article Switch**: Updated `config.py` to set `RAG_Anything` as the default `ACTIVE_ARTICLE`.
- **RAG Tuning**: Added article-specific overrides for `RAG_Anything` in `config.py`, enabling `langextract` metadata and entity filtering by default.
- **Query Updates**: Updated the active test query for the `RAG_Anything` paper in `queries.py`.

## [December 22, 2025]

### 1. RAG Factory & Architectural Refactoring

- **Centralized RAG Factory**: Introduced `rag_factory.py` as the single source of truth for building query engines and tools. This eliminates code duplication between `langextract_simple.py` and `minerU.py`.
- **Database Initialization Refactoring**: Introduced `get_storage_contexts()` in both `minerU.py` and `langextract_simple.py` to standardize how Milvus and MongoDB stores are initialized and verified.
- **Split-Brain Protection**: Integrated `handle_split_brain_state()` into the core ingestion pipeline. This ensures that if any part of the storage (Milvus Vector, Mongo Vector, or Mongo Summary) is missing, the system automatically resets and re-ingests all stores to maintain Node ID consistency.
- **Dynamic Metadata Filtering**: Implemented `DynamicFilterQueryEngine` which extracts semantic filters (via LangExtract) and named entities (via SpanMarker) for *every* sub-question. This significantly narrows the search space in Milvus for complex multi-part queries.
- **Lazy Initialization**: Refined `LazyQueryEngine` to defer resource-heavy tool initialization until the LLM actually selects the tool, reducing startup latency and API costs.
- **Visibility & Logging**: Added explicit terminal logging for "🔍 Extracted Query Filters" and "✓ Created X filters" to provide transparency into how metadata is being used during retrieval.

### 2. MinerU Pipeline Stabilization

- **Critical Bug Fixes**: Resolved 10+ issues in `minerU.py`, including:
    - Fixed a `RuntimeError` caused by nested async loops by applying `nest_asyncio.apply()`.
    - Corrected syntax errors in vector index creation and missing function implementations.
    - Fixed `NameError` for undefined variables like `query` and `page_filter_verbose`.
- **Deterministic Section Retrieval**: Restored and optimized `build_section_index_mineru` to allow exact mapping of section names (e.g., "CASE STUDIES") to page ranges, bypassing vector search for known sections.
- **Environment Robustness**: Improved `.env` loading and added validation for `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` at script startup.

### 3. Migration to MinerU as Primary Parser

- **Main Script Promotion**: Promoted `minerU.py` to the primary indexing and RAG entry point, replacing `llamaparse.py`.
- **Code Archival**: Moved the original `llamaparse.py` to [archive/llamaparse.py](archive/llamaparse.py) to maintain a clean root directory while preserving the legacy pipeline.
- **Metadata Stripping**: Integrated `MetadataStripperPostprocessor` in `utils.py` to strip large parser metadata (OCR/Layout data) before Milvus insertion, preventing token serialization blowups.
- **Documentation Update**: Updated [README.md](README.md) and internal documentation to reflect the shift to MinerU as the default parsing engine.
- **Pipeline Simplification**: Removed dual-parser logic from the main script to focus on the high-performance MinerU/MLX pipeline, improving maintainability.

### 2. Article Standardization & RAG Tuning

- **Article Naming**: Standardized article keys across `config.py`, `README.md`, and `README_GUIDE.md` (e.g., `attention_all`, `RAG_Anything`, `How_to_do_great_work`).
- **RAG Parameter Tuning**: Optimized default retrieval parameters in `config.py`:
    - Increased `chunk_overlap` to `128` for better context continuity.
    - Reduced fusion and rerank counts (`similarity_top_k_fusion=35`, `rerank_top_n=15`) to improve response speed.
    - Disabled `num_nodes` context expansion by default to focus on high-precision retrieval.
- **Configuration Centralization**: Refactored `minerU.py` to pull all retrieval and debug settings (e.g., `page_filter_verbose`) directly from `config.py`, ensuring system-wide consistency.
- **Default Article**: Switched the default `ACTIVE_ARTICLE` to `paul_graham_essay`.

### 3. LangExtract Pipeline Modernization

- **Sub-Question Generation**: Replaced `GuidanceQuestionGenerator` with `LLMQuestionGenerator` in `langextract_simple.py` for more reliable sub-query decomposition, eliminating dependencies on the `guidance` library.
- **Retry Mechanism**: Added a robust retry loop (3 attempts) for sub-question generation to handle transient JSON parsing failures from the LLM.
- **Metadata Resilience**: Updated node retrieval logic to use `.get()` for metadata keys, ensuring compatibility with both `'page'` (MinerU) and `'source'` (LlamaIndex) metadata schemas.
- **Enhanced Debugging**: Integrated `tiktoken` for precise token counting of retrieved context and added final response printing to the terminal output.

### 4. Centralized Query Management

- **New Query Repository**: Created `queries.py` to centralize all test queries for different articles, allowing for easy switching between prompts via comment/uncomment.
- **Dynamic Query Selection**: Integrated `queries.py` with `config.py` via a global `QUERY` variable that automatically tracks the `ACTIVE_ARTICLE`.
- **RAG Script Integration**: Updated `minerU.py` and `langextract_simple.py` to import and use the global `QUERY` from `config.py`, ensuring consistency across different RAG implementations.
- **Configuration Cleanup**: Removed legacy `sample_queries` from `ARTICLE_CONFIGS` in `config.py` to reduce clutter and maintain a single source of truth for active queries.

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
