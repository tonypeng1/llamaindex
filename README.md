## RAG using Llamaindex

RAG (Retrieval Augmented Generation) is a technique that combines the power of large language models (LLMs) with external data sources to enhance the accuracy and relevance of generated responses. 

A RAG that uses sub-question and tool-selecting query engines from llamaindex is demonstrated to query the article "What I Worked On" by Paul Graham.

## RAG Features
This RAG has the following features:

1. Use Milvus and MongoDB as the vector database and the document store, respectively. This code prevents duplicate database entries by checking for existing data before parsing the article and adding it to the database.
2. Use a sub-question query engine with 3 query engine tools: one for answering a query about specific information (keyphrase_tool), one for a query that requires the information on certain pages (page_filter_tool), and finally one for a query that needs a full-context summary (summary_tool). Details about these tools are discussed in the next section.
3. The keyphrase_tool and the page_filter_tool use a fusion retriever that combines a vector retriever and a BM25 retriever.
4. The noise of the BM25 retriever used in the keyphrase_tool is reduced by first extracting the key phrase of a query using the model KeyBERT, then using another BM25 retriever to retrieve the relevant text nodes with only the key phrase as the query (rather than the original full query), and finally defining the BM25 retriever with only these text nodes (rather than using all the nodes from the document store).
5. A Named Entity Recognition (NER) model extracts the named entities in each node.
6. The nodes retrieved by the fusion retriever are post-processed by a reranker, a previous-next postprocessor (to include all the adjacent nodes on a page), and finally by a custom sorting postprocessor (to sort the nodes in ascending order based on page number and node occurrence within a page using the data start_char_idx).

## File Structure

For this demonstration, 3 files are used:
1. llama_bm25_simple.py,
2. utility_simple.py, and
3. database_operation_simple.py.

The file "llama_bm25_simple.py" is the main file that uses the functions defined in the other two files.

## Requirement File
The requirement.txt file contains the necessary packages for this demonstration.

```
llama-index==0.11.7
llama-index-embeddings-huggingface==0.3.1
llama-index-storage-docstore-mongodb==0.2.0
llama-index-vector-stores-milvus==0.2.3
llama-index-retrievers-bm25==0.3.0
motor==3.5.1
pymilvus==2.4.6
PyMuPDF==1.24.10
keybert==0.8.5
llama-index-llms-mistralai==0.2.2
llama-parse==0.5.2
llama-index-llms-anthropic==0.3.1
llama-index-multi-modal-llms-anthropic==0.2.2
llama-index-postprocessor-flag-embedding-reranker==0.2.0
FlagEmbedding==1.2.11
llama-index-postprocessor-colbert-rerank==0.2.1
llama-index-question-gen-guidance==0.2.0
llama-index-extractors-entity==0.2.1
transformers==4.40.2
span-marker==1.5.0
```

Please note that there is a need to downgrade the transformers package to version 4.40.2, as the latest version is not compatible with the llamaindex packages.

## Article Directory
The .pdf file of Paull Graham's article “What I Worked On”, which is not included in this repository, is located at the directory:

```
./data/paul_graham/paul_graham_essay.pdf
```

This article can be found [here](https://drive.google.com/file/d/1YzCscCmQXn2IcGS-omcAc8TBuFrpiN4-/view?usp=sharing).

## Database and Collection Names
The database name (includes article name and parsing method) and collection/namespace name (includes embedding model, chuck size, overlap size, and metadata if available) in both the Milvus vector database and the MongoDB document store are:

- database name: "paul_graham_sentence_splitter"
- collection/namespace name: "openai_embedding_3_small_chunk_size_512_chunk_overlap_128_metadata_entity"

There is one more collection/namespace in the MongoDB document store, which is used to store the summary:

- collection/namespace name: "openai_embedding_3_small_chunk_size_512_chunk_overlap_128_summary_metadata_entity"

## Model API Key
The API keys for OpenAI and Anthropic are stored in the .env file, which is not included in this repository. The .evn file looks like:

```
OPENAI_API_KEY = your_openai_api_key
ANTHROPIC_API_KEY = your_anthropic_api_key
```

## Examples of Queries and Answers

4 sets of queries and answer that illustrate the capability of this RAG can be found in this [Medium artile](https://medium.com/@tony3t3t/rag-with-sub-question-and-tool-selecting-query-engines-using-llamaindex-05349cb4120c).

## Detailed Introduction

### Overview

`llama_bm25_simple.py` is an advanced Retrieval-Augmented Generation (RAG) system that implements a sophisticated multi-tool query engine for question-answering over PDF documents. The script combines vector-based semantic search with BM25 keyword-based retrieval using a sub-question decomposition approach to answer complex user queries about document content.

### Core Functionality

The script processes PDF documents through the following workflow:

1. **Document Loading & Parsing**: Loads PDF documents and splits them into manageable chunks using sentence-based splitting with configurable chunk sizes and overlaps
2. **Dual Storage Architecture**: Stores document embeddings in Milvus (vector database) and document metadata in MongoDB for efficient hybrid retrieval
3. **Multi-Tool Query System**: Routes queries to specialized tools based on query characteristics:
   - **Summary Tool**: For high-level document overviews and table of contents generation
   - **Keyphrase Tool**: For specific factual queries using BM25 keyword extraction combined with semantic search
   - **Page Filter Tool**: For queries targeting specific page numbers or ranges
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
- **Semantic Search**: Dense vector embeddings capture conceptual similarity
- **Keyword Search**: BM25 algorithm ensures precise keyword matching
- **Fusion Ranking**: Reciprocal rank fusion merges results from both approaches
- **Neural Re-ranking**: ColBERT provides fine-grained relevance scoring

This multi-stage retrieval pipeline ensures both broad conceptual understanding and precise factual accuracy, making it suitable for complex document Q&A tasks where users may ask questions ranging from high-level summaries to specific detail extraction.


