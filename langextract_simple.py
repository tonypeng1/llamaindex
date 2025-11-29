"""
LlamaIndex RAG Implementation with Flexible Metadata Extraction

This script implements a sophisticated RAG (Retrieval Augmented Generation) system
using LlamaIndex with multiple metadata extraction options:

Metadata Extraction Options:
----------------------------
1. None (Basic):
   - No metadata extraction
   - Fastest option
   - Simple chunking with SentenceSplitter only
   - Best for: Quick testing, simple documents

2. "entity" (EntityExtractor):
   - Uses HuggingFace span-marker model
   - Fast and free (local inference)
   - Extracts basic named entities (PER, ORG, LOC)
   - Device: MPS (Apple Silicon) or CPU
   - Best for: Standard entity recognition needs

3. "langextract" (LangExtract):
   - Uses Google's LangExtract with OpenAI GPT-4
   - Slow and paid (API calls)
   - Rich structured metadata (concepts, advice, experiences, etc.)
   - Requires: OPENAI_API_KEY environment variable
   - Best for: Deep semantic understanding, complex queries

4. "both" (EntityExtractor + LangExtract):
   - Combines both extractors
   - Slowest but most comprehensive
   - Preserves both entity and semantic metadata
   - Best for: Maximum metadata richness

Usage:
------
1. Set the `metadata` variable to your desired option
2. If using "langextract" or "both", set `schema_name` to your schema
3. Ensure required API keys are set in environment variables
4. Run the script to process documents with your chosen metadata extraction

Database Structure:
------------------
- Milvus: Vector index storage
- MongoDB: Document store for both vector and summary nodes
- Collections are named based on metadata extraction method used
"""

# Disable tokenizers parallelism warning (must be set before importing any tokenizers)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

import json
import nest_asyncio
import sys
import logging
# Suppress langextract and kor warnings
logging.getLogger("langextract").setLevel(logging.ERROR)
logging.getLogger("kor").setLevel(logging.ERROR)

from pathlib import Path
from typing import List, Optional

# Fix sys.excepthook error by ensuring a clean exception hook
sys.excepthook = sys.__excepthook__

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

from llama_index.core import (
                        Settings,
                        VectorStoreIndex,
                        )
from llama_index.core.callbacks import (
                        CallbackManager,
                        LlamaDebugHandler,
                        )
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import (
                        SentenceSplitter,
                        )
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import (
                        QueryEngineTool,
                        )
from llama_index.core.vector_stores import MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.extractors.entity import EntityExtractor
from llama_index.llms.anthropic import Anthropic
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.readers.file import (
                        PyMuPDFReader,
                        )
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.question_gen.guidance import GuidanceQuestionGenerator
from llama_index.core.prompts.guidance_utils import convert_to_handlebars
from guidance.models import OpenAI as GuidanceOpenAI

from utils import (
                get_article_link, 
                get_database_and_sentence_splitter_collection_name,
                get_fusion_tree_keyphrase_sort_detail_tool_simple,
                get_fusion_tree_page_filter_sort_detail_engine,
                get_summary_storage_context,
                get_summary_tree_detail_tool,
                get_vector_store_docstore_and_storage_context,
                )
from db_operation import (
                check_if_milvus_database_collection_exist,
                check_if_mongo_database_namespace_exist,
                handle_split_brain_state
                )
from langextract_integration import (
                enrich_nodes_with_langextract,
                print_sample_metadata
                )


def print_metadata_extraction_info():
    """
    Print information about available metadata extraction options.
    Helps users understand which option to choose.
    """
    info = """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                    METADATA EXTRACTION OPTIONS                             ║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║                                                                            ║
    ║  Option: None (Basic)                                                      ║
    ║  ├─ Speed: ⚡⚡⚡ Very Fast                                                 ║
    ║  ├─ Cost: FREE                                                             ║
    ║  ├─ Metadata: Basic (page numbers, file info only)                         ║
    ║  └─ Best for: Quick testing, simple documents                              ║
    ║                                                                            ║
    ║  Option: "entity" (EntityExtractor)                                        ║
    ║  ├─ Speed: ⚡⚡ Fast                                                       ║
    ║  ├─ Cost: FREE (local model)                                               ║
    ║  ├─ Metadata: Named entities (PER, ORG, LOC, etc.)                         ║
    ║  ├─ Model: span-marker-bert-base-multilingual-cased-multinerd             ║
    ║  ├─ Device: MPS (Apple Silicon) or CPU                                     ║
    ║  └─ Best for: Standard entity recognition                                  ║
    ║                                                                            ║
    ║  Option: "langextract" (LangExtract)                                       ║
    ║  ├─ Speed: ⚡ Slow (API calls)                                             ║
    ║  ├─ Cost: PAID (OpenAI API usage)                                          ║
    ║  ├─ Metadata: Rich structured metadata                                     ║
    ║  │   • Concepts (programming, philosophy, business, etc.)                  ║
    ║  │   • Advice (strategic, tactical, practical, etc.)                       ║
    ║  │   • Experiences (early career, success, challenges, etc.)               ║
    ║  │   • Entities (people, organizations, products)                          ║
    ║  │   • Time references (years, decades, periods)                           ║
    ║  ├─ Requirements: OPENAI_API_KEY in environment                            ║
    ║  └─ Best for: Deep semantic understanding, complex queries                 ║
    ║                                                                            ║
    ║  Option: "both" (EntityExtractor + LangExtract)                            ║
    ║  ├─ Speed: ⚡ Slowest (both extractors)                                    ║
    ║  ├─ Cost: PAID (OpenAI API usage)                                          ║
    ║  ├─ Metadata: Most comprehensive (entities + semantic metadata)            ║
    ║  └─ Best for: Maximum metadata richness                                    ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(info)


def print_current_configuration(metadata, schema_name, chunk_size, chunk_overlap, use_entity_filtering, 
                               similarity_top_k_fusion, num_queries, fusion_top_n, rerank_top_n, num_nodes):
    """
    Print the current configuration settings for the RAG system.
    
    Parameters:
    metadata (Optional[str]): The metadata extraction method being used
    schema_name (str): The LangExtract schema name
    chunk_size (int): The chunk size for text splitting
    chunk_overlap (int): The chunk overlap for text splitting
    use_entity_filtering (bool): Whether entity filtering is enabled
    similarity_top_k_fusion (int): Number of nodes to retrieve in fusion retrieval
    num_queries (int): Number of queries for fusion retrieval
    fusion_top_n (int): Top N nodes after fusion
    rerank_top_n (int): Top N nodes after reranking
    num_nodes (int): Number of nodes for PrevNextNodePostprocessor
    """
    print(f"\n📊 Current Configuration:")
    print(f"   Metadata Extraction: {metadata if metadata else 'None (Basic)'}")
    if metadata in ["langextract", "both"]:
        print(f"   LangExtract Schema: {schema_name}")
    print(f"   Chunk Size: {chunk_size}")
    print(f"   Chunk Overlap: {chunk_overlap}")
    if metadata in ["entity", "langextract", "both"]:
        print(f"   Entity Filtering: {'✓ Enabled' if use_entity_filtering else '✗ Disabled'}")
    print(f"\n   Fusion Tree & Reranker:")
    print(f"   ├─ Similarity Top K: {similarity_top_k_fusion}")
    print(f"   ├─ Number of Queries: {num_queries}")
    print(f"   ├─ Fusion Top N: {fusion_top_n}")
    print(f"   ├─ Rerank Top N: {rerank_top_n}")
    print(f"   └─ Prev/Next Nodes: {num_nodes}")
    print(f"\n{'='*80}\n")


def load_document_pdf(doc_link) -> List:
    """
    This function loads a PDF document from a given link and returns it as a 
    list of documents.

    Parameters:
    doc_link (str): The link to the PDF document.

    Returns:
    list: A list of documents extracted from the PDF.
    """
    loader = PyMuPDFReader()
    docs0 = loader.load(file_path=Path(doc_link))

    return docs0


def get_nodes_from_document_sentence_splitter(
        _document: List, 
        _chunk_size: int,
        _chunk_overlap: int
        ) -> List:
    """
    This function splits a document into nodes based on sentence boundaries.

    Parameters:
    _document (List): A list of documents to be split into nodes.
    _chunk_size (int): The size of each chunk.
    _chunk_overlap (int): The amount of overlap between chunks.

    Returns:
    list: A list of nodes, where each node is a chunk of the document.
    """

    # create the sentence spitter node parser
    node_parser = SentenceSplitter(
                                chunk_size=_chunk_size,
                                chunk_overlap=_chunk_overlap
                                )
    
    # _nodes = node_parser.get_nodes_from_documents([_document])
    _nodes = node_parser.get_nodes_from_documents(_document)

    return _nodes


def get_nodes_from_document_sentence_splitter_entity_extractor(
        _document: List, 
        _chunk_size: int,
        _chunk_overlap: int
        ) -> List:
    """
    This function splits a document into nodes based on sentence boundaries.

    Parameters:
    _document (List): A list of documents to be split into nodes.
    _chunk_size (int): The size of each chunk.
    _chunk_overlap (int): The amount of overlap between chunks.

    Returns:
    list: A list of nodes, where each node is a chunk of the document.
    """

    entity_extractor = EntityExtractor(
        model_name="lxyuan/span-marker-bert-base-multilingual-cased-multinerd",
        prediction_threshold=0.5,
        label_entities=True,
        # device="cpu",
        device="mps",
        # entity_map=entity_map,
    )

    # create the sentence spitter node parser
    node_parser = SentenceSplitter(
                                chunk_size=_chunk_size,
                                chunk_overlap=_chunk_overlap
                                )

    transformations = [node_parser, entity_extractor]
    pipeline = IngestionPipeline(transformations=transformations)

    _nodes_entity = pipeline.run(documents=_document)

    for n in _nodes_entity:
        print(n.metadata)

    return _nodes_entity


def load_document_nodes_sentence_splitter(
                                _article_link: str,
                                _chunk_size: int,
                                _chunk_overlap: int,
                                _metadata: Optional[str] = None,
                                _schema_name: str = "paul_graham_detailed"
                                ) -> List:  
    """
    This function loads a document from a given link, splits it into nodes based on sentences,
    and optionally extracts metadata from the nodes.

    Parameters:
    _article_link (str): The URL of the document to load.
    _chunk_size (int): The maximum size of each node.
    _chunk_overlap (int): The number of words that should overlap between consecutive nodes.
    _metadata (Optional[str], optional): Metadata extraction method:
        - "entity": Use EntityExtractor (fast, free, basic entity extraction)
        - "langextract": Use LangExtract (slow, paid, rich structured metadata)
        - "both": Use both extractors (EntityExtractor + LangExtract)
        - None: No metadata extraction (basic chunking only)
        Defaults to None.
    _schema_name (str): The LangExtract schema to use (only relevant for "langextract" or "both")
        Defaults to "paul_graham_detailed".

    Returns:
    List: A list of nodes, where each node is a dictionary containing the text of the node and optionally,
          the extracted metadata.
    """

    # Load and parse document
    _document = load_document_pdf(_article_link)

    # Process based on metadata extraction method
    if _metadata == "entity":
        # Use EntityExtractor only (fast, free)
        _nodes = get_nodes_from_document_sentence_splitter_entity_extractor(
            _document,
            _chunk_size,
            _chunk_overlap
            )
    elif _metadata == "langextract":
        # Use LangExtract only (slow, paid, rich metadata)
        _nodes = get_nodes_from_document_sentence_splitter(
            _document,
            _chunk_size,
            _chunk_overlap
            )
        # Enrich with LangExtract metadata
        _nodes = enrich_nodes_with_langextract(
            _nodes,
            schema_name=_schema_name,
            verbose=True
        )
        # Print sample metadata for verification
        print_sample_metadata(_nodes, num_samples=3)
        
    elif _metadata == "both":
        # Use both extractors (EntityExtractor + LangExtract)
        print("\n🔄 Using BOTH extractors: EntityExtractor + LangExtract")
        print("   Step 1: Running EntityExtractor (fast)...")
        _nodes = get_nodes_from_document_sentence_splitter_entity_extractor(
            _document,
            _chunk_size,
            _chunk_overlap
            )
        print("   Step 2: Running LangExtract (slow, API-based)...")
        # Enrich with LangExtract metadata (preserves EntityExtractor metadata)
        _nodes = enrich_nodes_with_langextract(
            _nodes,
            schema_name=_schema_name,
            verbose=True
        )
        # Print sample metadata for verification
        print_sample_metadata(_nodes, num_samples=3)
        
    else:
        # No metadata extraction (basic chunking only)
        _nodes = get_nodes_from_document_sentence_splitter(
            _document,
            _chunk_size,
            _chunk_overlap
            )
    
    return _nodes


def get_fusion_tree_page_filter_sort_detail_tool_simple(
    _query_str: str, 
    _reranker: ColbertRerank,
    _vector_docstore: MongoDocumentStore,
    *,
    verbose: bool = False,
    ) -> QueryEngineTool:
    """
    This function generates a QueryEngineTool that extracts specific pages from a document store,
    retrieves the text nodes corresponding to those pages, and then uses these nodes to create a fusion tree.
    The fusion tree is then used to answer queries about the content of the specified pages.

    Parameters:
    _query_str (str): The query string from which to extract page numbers.
    _reranker (ColbertRerank): The reranker to use for the fusion tree.
    _vector_docstore (MongoDocumentStore): The document store containing the text nodes.

    Returns:
    QueryEngineTool: A tool that uses the fusion tree to answer queries about the specified pages.
    """

    query_text = (
    '## Instruction:\n'
    'Extract all page numbers from the user query. \n'
    'The page numbers are usually indicated by the phrases "page" or "pages" \n'
    'Return the page numbers as a list of strings, sorted in ascending order. \n'
    'Do NOT include "**Output:**" in your response. If no page numbers are mentioned, output ["1"]. \n'

    '## Examples:\n'
    '**Query:** "Give me the main events from page 1 to page 4." \n'
    '**Output:** ["1", "2", "3", "4"] \n'

    '**Query:** "Give me the main events in the first 6 pages." \n'
    '**Output:** ["1", "2", "3", "4", "5", "6"] \n'

    '**Query:** "Summarize pages 10-15 of the document." \n'
    '**Output:** ["10", "11", "12", "13", "14", "15"] \n'

    '**Query:** "What are the key findings on page 2?" \n'
    '**Output:** ["2"] \n'

    '**Query:** "What is mentioned about YC (Y Combinator) on pages 19 and 20?" \n'
    '**Output:** ["19", "20"] \n'

    '**Query:** "What are the lessons learned by the author at the company Interleaf?" \n'
    '**Output:** ["1"] \n'

    '## Now extract the page numbers from the following query: \n'

    '**Query:** {query_str} \n'
    )

    # Need to print "1" if no page numbers are mentioned so that this code can run correctly

    prompt = PromptTemplate(query_text)
    _page_numbers = llm.predict(prompt=prompt, query_str=_query_str)
    _page_numbers = json.loads(_page_numbers)  # Convert the string to a list of string
    if verbose:
        print(f"Page_numbers in page filter: {_page_numbers}")

    # Get text nodes from the vector docstore that match the page numbers
    _text_nodes = []
    for _, node in _vector_docstore.docs.items():
        if node.metadata['source'] in _page_numbers:
            _text_nodes.append(node) 

    node_length = len(vector_docstore.docs)
    if verbose:
        print(f"Node length in docstore: {node_length}")
        print(f"Text nodes retrieved from docstore length is: {len(_text_nodes)}")
        for i, n in enumerate(_text_nodes):
            print(f"Item {i+1} of the text nodes retrieved from docstore is page: {n.metadata['source']}")
    
    _vector_filter_retriever = vector_index.as_retriever(
                                    similarity_top_k=node_length,
                                    filters=MetadataFilters.from_dicts(
                                        [{
                                            "key": "source", 
                                            "value": _page_numbers,
                                            "operator": "in"
                                        }]
                                    )
                                )
    
    # Calculate the number of nodes retrieved from the vector index on these pages
    _fusion_top_n_filter = len(_text_nodes)
    _num_queries_filter = 1

    _fusion_tree_page_filter_sort_detail_engine = get_fusion_tree_page_filter_sort_detail_engine(
                                                                _vector_filter_retriever,
                                                                _fusion_top_n_filter,
                                                                _text_nodes,
                                                                _num_queries_filter,
                                                                _reranker,
                                                                _vector_docstore,
                                                                _page_numbers
                                                                )
    
    _fusion_tree_page_filter_sort_detail_tool = QueryEngineTool.from_defaults(
                                                        name="page_filter_tool",
                                                        query_engine=_fusion_tree_page_filter_sort_detail_engine,
                                                        description=page_tool_description,
                                                        )

    return _fusion_tree_page_filter_sort_detail_tool


def create_custom_guidance_prompt() -> str:
    """
    Create a custom prompt template for GuidanceQuestionGenerator.
    
    This function constructs a prompt template that guides the question generation
    process for sub-question decomposition. The template includes:
    - A prefix explaining the task
    - Example 1: Financial comparison (default LlamaIndex example)
    - Example 2: Page-based content summarization
    - A suffix for the actual query
    
    Returns:
    str: A Handlebars-formatted prompt template string for use with GuidanceQuestionGenerator.
    """
    # Write in Python format string style, then convert to Handlebars
    PREFIX = """\
    Given a user question, and a list of tools, output a list of relevant sub-questions \
    in json markdown that when composed can help answer the full user question:

    """

    # Default example from LlamaIndex
    EXAMPLE_1 = """\
    # Example 1
    <Tools>
    ```json
    [
    {{
        "name": "uber_10k",
        "description": "Provides information about Uber financials for year 2021"
    }},
    {{
        "name": "lyft_10k",
        "description": "Provides information about Lyft financials for year 2021"
    }}
    ]
    ```

    <User Question>
    Compare and contrast the revenue growth and EBITDA of Uber and Lyft for year 2021

    <Output>
    ```json
    {{
    "items": [
        {{"sub_question": "What is the revenue growth of Uber", "tool_name": "uber_10k"}},
        {{"sub_question": "What is the EBITDA of Uber", "tool_name": "uber_10k"}},
        {{"sub_question": "What is the revenue growth of Lyft", "tool_name": "lyft_10k"}},
        {{"sub_question": "What is the EBITDA of Lyft", "tool_name": "lyft_10k"}}
    ]
    }}
    ```

    """

    # Tailored example for page-based queries
    EXAMPLE_2 = """\
    # Example 2
    <Tools>
    ```json
    [
    {{
        "name": "page_filter_tool",
        "description": "Perform a query search over the page numbers mentioned in the query"
    }}
    ]
    ```

    <User Question>
    Summarize the content from pages 20 to 22 in the voice of the author by NOT retrieving the text verbatim

    <Output>
    ```json
    {{
    "items": [
        {{"sub_question": "Summarize the content on page 20 in the voice of the author by NOT retrieving the text verbatim", "tool_name": "page_filter_tool"}},
        {{"sub_question": "Summarize the content on page 21 in the voice of the author by NOT retrieving the text verbatim", "tool_name": "page_filter_tool"}},
        {{"sub_question": "Summarize the content on page 22 in the voice of the author by NOT retrieving the text verbatim", "tool_name": "page_filter_tool"}}
    ]
    }}
    ```

    """

    SUFFIX = """\
    # Example 3
    <Tools>
    ```json
    {tools_str}
    ```

    <User Question>
    {query_str}

    <Output>
    """

    # Combine and convert to Handlebars format
    custom_guidance_prompt = convert_to_handlebars(PREFIX + EXAMPLE_1 + EXAMPLE_2 + SUFFIX)
    # Alternative with only one example:
    # custom_guidance_prompt = convert_to_handlebars(PREFIX + EXAMPLE_1 + SUFFIX)
    
    return custom_guidance_prompt


class LazyQueryEngine:
    """Instantiate the underlying query engine only when first queried."""

    # This wrapper defers tool creation until the sub-question engine actually calls
    # the query, preventing eager page-filter initialization (and its logging) when
    # the tool is merely registered but not used.

    def __init__(self, factory):
        self._factory = factory
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = self._factory()

    def query(self, *args, **kwargs):
        self._ensure_engine()
        return self._engine.query(*args, **kwargs)

    async def aquery(self, *args, **kwargs):
        self._ensure_engine()
        if hasattr(self._engine, "aquery"):
            return await self._engine.aquery(*args, **kwargs)
        raise NotImplementedError("Underlying query engine does not support async queries.")

    def __getattr__(self, item):
        self._ensure_engine()
        return getattr(self._engine, item)


def build_page_filter_query_engine():
    tool = get_fusion_tree_page_filter_sort_detail_tool_simple(
        query_str,
        colbert_reranker,
        vector_docstore,
        verbose=page_filter_verbose,
    )
    return tool.query_engine


# Set LLM and embedding models
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
llm = Anthropic(
    model="claude-sonnet-4-0",
    temperature=0.0,
    max_tokens=2000,
    api_key=ANTHROPIC_API_KEY,
    )
Settings.llm = llm

embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
Settings.embed_model = embed_model
embed_model_dim = 1536  # for text-embedding-3-small
embed_model_name = "openai_embedding_3_small"

# Create debug handler
llama_debug = LlamaDebugHandler(print_trace_on_end=False)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

# Create article link
article_directory = "paul_graham"
article_name = "paul_graham_essay.pdf"
# article_name = "How_to_do_great_work.pdf"

article_link = get_article_link(article_directory,
                                article_name
                                )

# Create database and collection names
chunk_method = "sentence_splitter"
# chunk_size = 512
# chunk_overlap = 128
chunk_size = 256
chunk_overlap = 64

# Metadata extraction options:
# None, "entity", "langextract", and "both"

# metadata = "both"
metadata = "entity"
# metadata = "langextract"
# metadata = None

# LangExtract schema (only used when metadata is "langextract" or "both")
# Available schemas: "paul_graham_detailed", "paul_graham_simple"
schema_name = "paul_graham_detailed"

# Entity-based filtering configuration
use_entity_filtering = True
# use_entity_filtering = False 

# Page filter debug logging
page_filter_verbose = True  # Set to False when you want quieter runs

# Fusion tree and reranker configuration
similarity_top_k_fusion = 36
num_queries = 1
fusion_top_n = 32
rerank_top_n = 24
# rerank_top_n = 12
num_nodes = 0 # For PrevNextNodePostprocessor

# print metadata extraction info and fusion tree and reranker configurations
print_current_configuration(metadata, schema_name, chunk_size, chunk_overlap, use_entity_filtering,
                           similarity_top_k_fusion, num_queries, fusion_top_n, rerank_top_n, num_nodes)

# metadata is an optional parameter, will include it if it is not None                                              )
(database_name, 
collection_name_vector,
collection_name_summary) = get_database_and_sentence_splitter_collection_name(
                                                            article_directory, 
                                                            article_name,
                                                            chunk_method, 
                                                            embed_model_name, 
                                                            chunk_size,
                                                            chunk_overlap,
                                                            metadata
                                                            )

# Initiate Milvus and MongoDB database
uri_milvus = "http://localhost:19530"
uri_mongo = "mongodb://localhost:27017/"

# Check if the vector index has already been saved to Milvus database. 
# If not, set save_index_vector to True.
save_index_vector = check_if_milvus_database_collection_exist(uri_milvus, 
                                                       database_name, 
                                                       collection_name_vector
                                                       )

# Check if the vector document has already been saved to MongoDB database.
# If not, set save_document_vector to True.
add_document_vector = check_if_mongo_database_namespace_exist(uri_mongo, 
                                                       database_name, 
                                                       collection_name_vector
                                                       )

# Check if the summary document has already been saved to MongoDB database.
add_document_summary = check_if_mongo_database_namespace_exist(uri_mongo, 
                                                       database_name, 
                                                       collection_name_summary
                                                       )

# --- FIX FOR SPLIT-BRAIN ID MISMATCH ---
# If ANY of the stores (Milvus Vector, Mongo Vector, or Mongo Summary) is missing,
# we must re-ingest ALL of them to ensure the Node IDs match across the entire system.
# This is crucial because SentenceSplitter generates non-deterministic IDs (different IDs on each run).

(save_index_vector, 
 add_document_vector, 
 add_document_summary) = handle_split_brain_state(
                        save_index_vector,
                        add_document_vector,
                        add_document_summary,
                        uri_milvus,
                        uri_mongo,
                        database_name,
                        collection_name_vector,
                        collection_name_summary
                        )

# Create vector store, vector docstore, and vector storage context
(vector_store,
 vector_docstore,
 storage_context_vector) = get_vector_store_docstore_and_storage_context(uri_milvus,
                                                                    uri_mongo,
                                                                    database_name,
                                                                    collection_name_vector,
                                                                    embed_model_dim
                                                                    )

# Create summary storage context
storage_context_summary = get_summary_storage_context(uri_mongo,
                                                    database_name,
                                                    collection_name_summary
                                                    )

# Load documnet nodes if we need to ingest (flags are synchronized now, all True or all False)
# metadata is an optional parameter, will use another function to parse the document
# if the metadata is not None.
if save_index_vector: 
    extracted_nodes = load_document_nodes_sentence_splitter(
                                                    article_link,
                                                    chunk_size,
                                                    chunk_overlap,
                                                    metadata,
                                                    schema_name
                                                    )

    # Create new index (ingests into Milvus and Mongo Vector Store)
    vector_index = VectorStoreIndex(
        nodes=extracted_nodes,
        storage_context=storage_context_vector,
        callback_manager=callback_manager,
        )
    
    # Save document nodes to Mongodb docstore at the server
    storage_context_vector.docstore.add_documents(extracted_nodes)
    
    # Save document nodes to Summary Store (separate context)
    # Note: VectorStoreIndex handles storage_context_vector, but we must manually add to summary store
    storage_context_summary.docstore.add_documents(extracted_nodes)

else:
    # Load from Milvus database
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context_vector
        )

summary_tool_description = (
            "Useful for a question that requires the full context of the entire document. "
            "DO NOT use this tool if user asks to summarize a specific section, topic, "
            "or event in a document. Use this tool when user asks to: 'Summarize the entire document' or "
            "'Give me an overview' or 'What is the main idea of the document?' or 'What is the document about?' "
            "or 'Create document outlines' or 'Create table of contents'. "
            )

summary_tool = get_summary_tree_detail_tool(
                                        summary_tool_description,
                                        storage_context_summary
                                        )

# query_str = "What was mentioned about Jessica from pages 17 to 22?"
# query_str = "What did Paul Graham do in 1980, in 1996 and in 2019?"
# query_str = "What did the author do after handing off Y Combinator to Sam Altman?"
# query_str = "What strategic advice is given about startups?"
# query_str = "Has the author been to Europe?"
# query_str = "What was mentioned about Jessica from pages 17 to 19?"
# query_str = "List all people mentioned in the document."
# query_str = "What experiences from the 1990s are described?"
# query_str = "What programming concepts are given in the document?"
# query_str = "Who are mentioned as colleagues in the document?"
query_str = "Does the author have any advice on relationships?"

# query_str = "Create table of contents for this article."

# query_str = "What did the author advice on choosing what to work on?"
# query_str = "Why morale needs to be nurtured and protected?" 
# query_str = "What are the contents from pages 26 to 29?"
# query_str = "What are the contents from pages 20 to 24 (one page at a time)?"
# query_str = ("What are the concise contents from pages 20 to 24 (one page at a time) in the voice of the author?"
#              )
# query_str = (
#     "Summarize the content from pages 1 to 5 (one page at a time) in the voice of the author by NOT retrieving the text verbatim."
#     )
# query_str = (
#     "Summarize the key takeaways from pages 1 to 5 (one page at a time) in a sequential order and in the voice of the author by NOT retrieving the text verbatim."
#     )
# query_str = (
#     "Summarize the key contents from pages 1 to 5 (one page at a time) in the voice of the author by NOT retrieving the text verbatim."
#     )

vector_store.client.load_collection(collection_name=collection_name_vector)

# Define reranker
colbert_reranker = ColbertRerank(
    top_n=rerank_top_n,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)

specific_tool_description = (
            "Useful for retrieving specific, precise, or targeted content from the document. "
            "Use this fuction to pinpoint the relevant information from the document "
            "when user seeks factual answer or a specific detail from the document, "
            "for example, when user uses interrogative words like 'what', 'who', 'where', "
            "'when', 'why', 'how', which may not require understanding the entire document "
            "to provide an answer."
            )

# fusion_keyphrase_tool: "Useful for retrieving SPECIFIC context from the document."
# Enable entity filtering when using entity metadata extraction AND use_entity_filtering is True
enable_entity_filtering = use_entity_filtering and metadata in ["entity", "langextract", "both"]

keyphrase_tool = get_fusion_tree_keyphrase_sort_detail_tool_simple(
                                                    vector_index,
                                                    vector_docstore,
                                                    similarity_top_k_fusion,
                                                    fusion_top_n,
                                                    query_str,
                                                    num_queries,
                                                    colbert_reranker,
                                                    specific_tool_description,
                                                    enable_entity_filtering=enable_entity_filtering,
                                                    metadata_option=metadata if metadata else None,
                                                    llm=llm,
                                                    num_nodes=num_nodes, # For PrevNextNodePostprocessor
                                                    )

page_tool_description = (
                "Perform a query search over the page numbers mentioned in the query. "
                "Use this function when user only need to retrieve information from specific PAGES, "
                "for example when user asks 'What happened on PAGE 19?' "
                "or 'What are the things mentioned on PAGES 19 and 20?' "
                "or 'Describe the contents from PAGE 1 to PAGE 4'. "
                "DO NOT GENERATE A SUB-QUESTION ASKING ABOUT ONE PAGE ONLY "
                "IF EQUAL TO OR MORE THAN 2 PAGES ARE MENTIONED IN THE QUERY. "
                )

# Use LazyQueryEngine to defer initialization until first use
lazy_page_filter_engine = LazyQueryEngine(build_page_filter_query_engine)

page_filter_tool = QueryEngineTool.from_defaults(
    name="page_filter_tool",
    query_engine=lazy_page_filter_engine,
    description=page_tool_description,
    )

CUSTOM_GUIDANCE_PROMPT = create_custom_guidance_prompt()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
question_gen = GuidanceQuestionGenerator.from_defaults(
                            prompt_template_str=CUSTOM_GUIDANCE_PROMPT,
                            guidance_llm=GuidanceOpenAI(
                                model="gpt-4o",
                                api_key=OPENAI_API_KEY,
                                echo=False),
                            verbose=True
                            )

tools=[
    keyphrase_tool,
    summary_tool,
    page_filter_tool
    ]

sub_question_engine = SubQuestionQueryEngine.from_defaults(
                                        question_gen=question_gen, 
                                        query_engine_tools=tools,
                                        verbose=True,
                                        )

response = None
try:
    response = sub_question_engine.query(query_str)
except Exception as e:
    print(f"Error getting json answer from LLM: {e}")

if response is not None:
    # Collect nodes with metadata (actual document nodes)
    document_nodes = []
    
    for i, n in enumerate(response.source_nodes):
        if bool(n.metadata): # the first few nodes may not have metadata (the LLM response nodes)
            # print(f"Item {i+1} of the source pages of response is page: {n.metadata['source']} \
            # (with score: {round(n.score, 3) if n.score is not None else None})")
            # Store node info for sequential output
            document_nodes.append({
                'page': n.metadata['source'],
                'text': n.text,
                'score': n.score
            })
        else:
            print(f"Item {i+1} question and response:\n{n.text}\n ")
    
    # Output sequential nodes with page numbers in a list
    if document_nodes:
        print("\n" + "="*80)
        print("SEQUENTIAL NODES WITH PAGE NUMBERS (sent to LLM for final answer):")
        print("="*80)
        for i, node_info in enumerate(document_nodes, 1):
            print(f"--- Node {i} | Page {node_info['page']} | Score: {round(node_info['score'], 3) if node_info['score'] is not None else 'N/A'} ---")
            # print(f"{node_info['text']}")
            # print("-" * 80)
        print("="*80 + "\n")
# Cleanup resources
try:
    if 'vector_store' in dir() and hasattr(vector_store, 'client'):
        vector_store.client.release_collection(collection_name=collection_name_vector)
        vector_store.client.close()
except:
    pass

# Suppress warnings and force immediate exit to avoid excepthook errors during Python shutdown
import warnings
warnings.filterwarnings('ignore')
os._exit(0)







