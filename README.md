# RAG using LlamaIndex

A Retrieval Augmented Generation (RAG) system that uses sub-question and tool-selecting query engines from LlamaIndex to query Paul Graham's article "What I Worked On".

## Quick Start

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
uv pip install -r requirements_arm64.txt  # For Apple Silicon Macs
# OR
uv pip install -r requirements.txt        # For x86 systems
```

> **Note:** You can still use regular `pip` if preferred, but `uv` is recommended for speed and reliability.

### Setup

1. **Download the article**: Get Paul Graham's "What I Worked On" ([download here](https://drive.google.com/file/d/1YzCscCmQXn2IcGS-omcAc8TBuFrpiN4-/view?usp=sharing)) and place it at:
   ```
   ./data/paul_graham/paul_graham_essay.pdf
   ```

2. **Database setup**: Ensure Milvus and MongoDB are running locally or accessible via network.

3. **API Keys**: Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

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
  - **Fusion**: Always combines both retrievers with 50/50 weighting for optimal results
- **KeyBERT Integration**: Reduces BM25 noise by extracting key phrases before retrieval
- **Advanced Post-processing**: Reranking, context expansion, and page-based sorting

## Metadata Extraction Options

The system supports four metadata extraction methods to suit different needs:

### 1. None (Basic) - Fast & Free
- Basic chunking with page numbers only
- **Use case**: Quick testing, simple documents
- **Speed**: ⚡⚡⚡ Very Fast (~10s for 30 pages)
- **Cost**: FREE

### 2. EntityExtractor - Fast & Free
- Named entity recognition (PERSON, ORGANIZATION, LOCATION, etc.)
- Uses HuggingFace span-marker model locally
- **Use case**: Standard entity recognition needs
- **Speed**: ⚡⚡ Fast (~30s for 30 pages)
- **Cost**: FREE

### 3. LangExtract - Slow & Paid
- Rich semantic metadata: concepts, advice, experiences, entities, time references
- Uses Google's LangExtract with OpenAI GPT-4
- **Use case**: Deep semantic understanding, complex queries
- **Speed**: ⚡ Slow (~15 min for 30 pages)
- **Cost**: ~$2 for 30 pages with GPT-4

### 4. Both - Most Comprehensive
- Combines EntityExtractor + LangExtract
- **Use case**: Maximum metadata richness
- **Speed**: ⚡ Slowest (~16 min for 30 pages)
- **Cost**: ~$2 for 30 pages

**Quick Configuration:**
```python
# In langextract_simple.py, set:
metadata = "entity"  # Options: None, "entity", "langextract", "both"
schema_name = "paul_graham_detailed"  # For LangExtract
use_entity_filtering = True  # Enable entity-based filtering (default: True)
```

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

`llama_bm25_simple.py` is an advanced Retrieval-Augmented Generation (RAG) system that implements a sophisticated multi-tool query engine for question-answering over PDF documents. The script combines vector-based semantic search with BM25 keyword-based retrieval using a sub-question decomposition approach to answer complex user queries about document content.

### Core Functionality

The script processes PDF documents through the following workflow:

1. **Document Loading & Parsing**: Loads PDF documents and splits them into manageable chunks using sentence-based splitting with configurable chunk sizes and overlaps
2. **Dual Storage Architecture**: Stores document embeddings in Milvus (vector database) and document metadata in MongoDB for efficient hybrid retrieval
3. **Multi-Tool Query System**: Routes queries to specialized tools based on query characteristics
4. **Sub-Question Decomposition**: Breaks down complex queries into simpler sub-questions that are routed to appropriate tools
5. **Advanced Re-ranking**: Uses ColBERT neural re-ranking to improve retrieval quality

### Key Imported Libraries

#### **LlamaIndex Core Components** (`llama_index.core`)
- **VectorStoreIndex**: Creates and manages vector-based document indexes for semantic search
- **Settings**: Configures global LLM and embedding model settings
- **CallbackManager & LlamaDebugHandler**: Provides debugging and tracing capabilities
- **IngestionPipeline**: Orchestrates document processing transformations
- **SentenceSplitter**: Splits documents into chunks at sentence boundaries
- **SubQuestionQueryEngine**: Decomposes complex queries into simpler sub-questions
- **QueryEngineTool**: Wraps query engines as tools for the sub-question engine
- **MetadataFilters**: Enables filtering retrieval results by metadata (e.g., page numbers)

#### **LLM & Embedding Models**
- **Anthropic**: Claude 3 Sonnet model for natural language understanding and generation
- **OpenAIEmbedding**: text-embedding-3-small model for converting text into 1536-dimensional vectors

#### **Specialized Processors**
- **EntityExtractor** (`llama_index.extractors.entity`): Extracts named entities (people, organizations, locations) from text using a multilingual BERT model
- **ColbertRerank** (`llama_index.postprocessor.colbert_rerank`): Neural re-ranking using ColBERT for improved retrieval accuracy
- **PrevNextNodePostprocessor** (from `utility_simple.py`): Retrieves adjacent context nodes for better answer completeness
- **PageSortNodePostprocessor** (from `utility_simple.py`): Sorts retrieved nodes by page number and position for coherent responses

#### **Retrieval Methods**
- **BM25Retriever** (`llama_index.retrievers.bm25`): Implements BM25 probabilistic keyword-based retrieval (Okapi BM25 algorithm)
- **QueryFusionRetriever** (from `utility_simple.py`): Combines vector and BM25 retrieval using reciprocal rank fusion

#### **Document Readers & Storage**
- **PyMuPDFReader** (`llama_index.readers.file`): Reads and extracts text from PDF files with page-level metadata
- **MilvusVectorStore** (`llama_index.vector_stores.milvus`): Interfaces with Milvus for vector similarity search
- **MongoDocumentStore** (`llama_index.storage.docstore.mongodb`): Stores document text and metadata in MongoDB

#### **Query Generation**
- **GuidanceQuestionGenerator** (`llama_index.question_gen.guidance`): Uses structured generation (Guidance library) with GPT-4o to decompose queries into focused sub-questions
- **KeyBERT** (from `utility_simple.py` via `keybert`): Extracts keyphrases from queries for enhanced BM25 retrieval

#### **Database Operations**
Custom utility functions (`database_operation_simple.py`) handle database management:
- Check existence of Milvus collections and MongoDB namespaces
- Create databases and count collection items
- Manage data persistence across sessions

#### **Custom Utilities**
Helper functions (`utility_simple.py`) provide:
- Prompt template customization for detailed responses
- Fusion retrieval engine construction
- Tool creation and configuration
- Page-based filtering and keyphrase extraction

### Technical Architecture

The system implements a **hybrid retrieval architecture** that combines:
- **Semantic Search**: Dense vector embeddings capture conceptual similarity (Milvus)
- **Keyword Search**: BM25 algorithm ensures precise keyword matching (MongoDB)
- **Fusion Ranking**: Reciprocal rank fusion merges results from both approaches (50/50 weighting)
- **Neural Re-ranking**: ColBERT provides fine-grained relevance scoring

#### Entity Filtering Architecture

When `use_entity_filtering = True` and metadata extraction includes entities:

1. **Entity Extraction**: Detects entities (people, orgs, locations) from the user query
2. **Parallel Retrieval**:
   - **BM25 Retriever**: 
     - Operates on MongoDB docstore
     - Uses KeyBERT keyphrase extraction
     - **NO entity filtering** (searches full docstore)
     - Ensures keyword-relevant results aren't missed
   - **Vector Retriever**:
     - Operates on Milvus vector index  
     - Uses semantic similarity
     - **WITH entity filtering** (only nodes mentioning entities)
     - Improves precision for entity-focused queries
3. **Fusion**: Always combines both retrievers with equal weighting
4. **Reranking**: ColBERT final ranking for optimal precision

This multi-stage retrieval pipeline ensures both broad conceptual understanding and precise factual accuracy, making it suitable for complex document Q&A tasks where users may ask questions ranging from high-level summaries to specific detail extraction.

## Database Access Patterns

This section outlines all instances in the codebase where MongoDB and Milvus databases are accessed, showing how the dual storage architecture operates.

### **MongoDB Database Access**

#### **1. Database Existence Checks** (`database_operation_simple.py`)

```python
def check_if_mongo_database_exists(uri, _database_name) -> bool:
    client = MongoClient(uri)
    db_names = client.list_database_names()  # ← MongoDB ACCESS
    client.close()
```

```python
def check_if_mongo_namespace_exists(uri, db_name, namespace) -> bool:
    client = MongoClient(uri)
    db = client[db_name]
    collection_names = db.list_collection_names()  # ← MongoDB ACCESS
```

#### **2. Document Storage Operations** (`llama_bm25_simple.py`)

```python
# Save document nodes to MongoDB docstore
if add_document_vector == True:
    storage_context_vector.docstore.add_documents(extracted_nodes)  # ← MongoDB WRITE

if add_document_summary == True:
    storage_context_summary.docstore.add_documents(extracted_nodes)  # ← MongoDB WRITE
```

#### **3. Document Retrieval for BM25** (`utility_simple.py`)

```python
# Iterate through all documents in MongoDB docstore
for _, node in _vector_docstore.docs.items():  # ← MongoDB READ
    if node.metadata['source'] in _page_numbers:
        _text_nodes.append(node)
```

```python
# Create BM25 retriever from MongoDB docstore
# NOTE: BM25 always operates on FULL docstore (no entity filtering)
bm25_retriever = BM25Retriever.from_defaults(
    similarity_top_k=similarity_top_n,
    docstore=vector_docstore,  # ← MongoDB READ (implicit)
)
```

**Important**: BM25 retriever always operates on the complete MongoDB docstore without entity filtering. This ensures keyword-relevant results are not missed even when entity filtering is enabled for the vector retriever.

#### **4. Summary Index Creation** (`utility_simple.py`)

```python
def get_summary_tree_detail_engine(storage_context_summary):
    extracted_nodes = list(storage_context_summary.docstore.docs.values())  # ← MongoDB READ
    summary_index = SummaryIndex(nodes=extracted_nodes)
```

#### **5. PrevNext Node Processing** (`utility_simple.py`)

```python
PrevNext = PrevNextNodePostprocessor(
    docstore=vector_docstore,  # ← MongoDB READ (for adjacent nodes)
    num_nodes=2,
    mode="both",
)
```

### **Milvus Database Access**

#### **1. Database Management** (`database_operation_simple.py`)

```python
def check_if_milvus_database_exists(uri, database_name) -> bool:
    connections.connect(uri=uri)  # ← Milvus CONNECTION
    db_names = db.list_database()  # ← Milvus READ
```

```python
def create_database_milvus(uri, db_name):
    connections.connect(uri=uri)  # ← Milvus CONNECTION
    db.create_database(db_name)  # ← Milvus WRITE
```

#### **2. Collection Operations** (`database_operation_simple.py`)

```python
def check_if_milvus_collection_exists(uri, db_name, collect_name) -> bool:
    client = MilvusClient(uri=uri, db_name=db_name)  # ← Milvus CONNECTION
    collect_names = client.list_collections()  # ← Milvus READ
```

```python
def milvus_collection_item_count(uri, database_name, collection_name) -> int:
    client = MilvusClient(uri=uri, db_name=database_name)  # ← Milvus CONNECTION
    client.load_collection(collection_name=collection_name)  # ← Milvus READ
    element_count = client.query(collection_name=collection_name, output_fields=["count(*)"])  # ← Milvus READ
```

#### **3. Vector Index Creation** (`llama_bm25_simple.py`)

```python
# Create new vector index (writes embeddings to Milvus)
if save_index_vector == True:
    vector_index = VectorStoreIndex(
        nodes=extracted_nodes,
        storage_context=storage_context_vector,  # ← Milvus WRITE
    )
```

#### **4. Vector Index Loading** (`llama_bm25_simple.py`)

```python
# Load existing vector index from Milvus
else:
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store  # ← Milvus READ
    )
```

#### **5. Vector Search Operations** (`llama_bm25_simple.py` & `utility_simple.py`)

```python
# Vector retrieval with metadata filtering
_vector_filter_retriever = vector_index.as_retriever(
    similarity_top_k=node_length,
    filters=MetadataFilters.from_dicts([{...}])  # ← Milvus READ
)
```

```python
# Standard vector retrieval
vector_retriever = vector_index.as_retriever(
    similarity_top_k=similarity_top_k_fusion,  # ← Milvus READ
)

scored_nodes = vector_retriever.retrieve(query_str)  # ← Milvus READ
```

#### **6. Collection Management** (`llama_bm25_simple.py`)

```python
# Load collection into memory for search
vector_store.client.load_collection(collection_name=collection_name_vector)  # ← Milvus READ

# Release collection from memory
vector_store.client.release_collection(collection_name=collection_name_vector)  # ← Milvus CLEANUP

# Close connection
vector_store.client.close()  # ← Milvus CLEANUP
```

#### **7. Fusion Retrieval** (`utility_simple.py`)

```python
# QueryFusionRetriever combines vector + BM25 retrieval
fusion_filter_retriever = QueryFusionRetriever(
    retrievers=[
        vector_filter_retriever,  # ← Milvus READ (implicit during retrieval)
        bm25_filter_retriever     # ← MongoDB READ (implicit during retrieval)
    ],
)
```

### **Summary of Access Patterns**

#### **MongoDB Usage**:
- **Configuration checks**: Database/namespace existence
- **Document storage**: Saving processed nodes
- **Text retrieval**: Full document content for BM25, context windows
- **Node relationships**: Adjacent node retrieval for context

#### **Milvus Usage**:
- **Configuration checks**: Database/collection existence  
- **Vector storage**: Saving embeddings during index creation
- **Semantic search**: Vector similarity retrieval
- **Metadata filtering**: Page-based and entity-based filtering
- **Collection management**: Load/release for memory optimization

#### **Dual Access (Fusion)**:
- **QueryFusionRetriever**: Simultaneously queries both databases
- **Hybrid retrieval**: Combines semantic (Milvus) + keyword (MongoDB) search
- **Results merging**: Uses reciprocal rank fusion to combine results

The architecture maintains **separation of concerns**: Milvus handles vector operations while MongoDB manages document text and metadata, with fusion mechanisms bridging both systems for comprehensive retrieval.
