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

import os
from dotenv import load_dotenv
import nest_asyncio
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Any

load_dotenv()

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress langextract and kor warnings
logging.getLogger("langextract").setLevel(logging.ERROR)
logging.getLogger("kor").setLevel(logging.ERROR)

# Fix sys.excepthook error by ensuring a clean exception hook
sys.excepthook = sys.__excepthook__

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Silence the verbose HTTP request logs from Anthropic/OpenAI
logging.getLogger("httpx").setLevel(logging.WARNING)

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
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.extractors.entity import EntityExtractor
from llama_index.llms.anthropic import Anthropic
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.readers.file import (
                        PyMuPDFReader,
                        )

from utils import (
                get_article_link, 
                get_database_and_sentence_splitter_collection_name,
                get_summary_storage_context,
                get_summary_tree_detail_tool,
                stitch_prev_next_relationships,
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
import rag_factory
from config import (
                get_active_config,
                get_rag_settings,
                print_article_summary,
                EMBEDDING_CONFIG,
                DATABASE_CONFIG,
                )


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
    
    _nodes = stitch_prev_next_relationships(_nodes)

    return _nodes


def get_storage_contexts(
    uri_milvus: str,
    uri_mongo: str,
    database_name: str,
    collection_name_vector: str,
    collection_name_summary: str,
    embed_model_dim: int
) -> Tuple[bool, bool, bool, Any, Any, Any, Any]:
    """
    Initializes storage contexts and checks for existing collections.
    """
    # Check if the vector index has already been saved to Milvus database.
    save_index_vector = check_if_milvus_database_collection_exist(uri_milvus, 
                                                           database_name, 
                                                           collection_name_vector
                                                           )

    # Check if the vector document has already been saved to MongoDB database.
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
    
    return (
        save_index_vector,
        add_document_vector,
        add_document_summary,
        vector_store,
        vector_docstore,
        storage_context_vector,
        storage_context_summary
    )


# Set LLM and embedding models
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
llm = Anthropic(
    model="claude-sonnet-4-0",
    temperature=0.0,
    max_tokens=2000,
    api_key=ANTHROPIC_API_KEY,
    )
Settings.llm = llm

embed_model = OpenAIEmbedding(model_name=EMBEDDING_CONFIG["model_name"])
Settings.embed_model = embed_model
embed_model_dim = EMBEDDING_CONFIG["dimension"]
embed_model_name = EMBEDDING_CONFIG["short_name"]

# Create debug handler
llama_debug = LlamaDebugHandler(print_trace_on_end=False)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

# =============================================================================
# Get ALL settings from config.py and set up variables
# NOTE: Set ACTIVE_ARTICLE in config.py to choose which document is processed
# =============================================================================

# Get ACTIVE_ARTICLE file name, directory, and other settings from config.py
article_config = get_active_config()
article_directory = article_config["directory"]
article_name = article_config["filename"]

article_link = get_article_link(article_directory,
                                article_name
                                )

# Get ALL RAG settings with any article-specific overrides
rag_settings = get_rag_settings()  # no parameter, uses ACTIVE_ARTICLE internally

# Create database and collection names
chunk_method = rag_settings["chunk_method"]
chunk_size = rag_settings["chunk_size"]
chunk_overlap = rag_settings["chunk_overlap"]

# Metadata extraction options:
# None, "entity", "langextract", and "both"
metadata = rag_settings["metadata"]

# LangExtract schema (from article config, only used when metadata is "langextract" or "both")
schema_name = article_config["schema"]

# Entity-based filtering configuration
use_entity_filtering = rag_settings["use_entity_filtering"]

# Note: Dynamic filtering (extracting filters per sub-question) is always used when
# entity filtering is enabled. This is more accurate for multi-part questions like
# "What did X do in 1980, 1996, and 2019?"

# Page filter debug logging
page_filter_verbose = rag_settings["page_filter_verbose"]

# Fusion tree and reranker configuration
similarity_top_k_fusion = rag_settings["similarity_top_k_fusion"]
num_queries = rag_settings["num_queries"]
fusion_top_n = rag_settings["fusion_top_n"]
rerank_top_n = rag_settings["rerank_top_n"]
num_nodes = rag_settings["num_nodes"]

# Print article configuration summary (ALL settings)
print_article_summary()

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
uri_milvus = DATABASE_CONFIG["milvus_uri"]
uri_mongo = DATABASE_CONFIG["mongo_uri"]

# Initialize storage contexts and check for existing collections
(save_index_vector,
 add_document_vector,
 add_document_summary,
 vector_store,
 vector_docstore,
 storage_context_vector,
 storage_context_summary) = get_storage_contexts(
                                                uri_milvus,
                                                uri_mongo,
                                                database_name,
                                                collection_name_vector,
                                                collection_name_summary,
                                                embed_model_dim
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
# query_str = "What was mentioned about Jessica from pages 17 to 22? Please cite page numbers in your answer."
query_str = "What did Paul Graham do in 1980, in 1996 and in 2019?"
# query_str = "What did the author do after handing off Y Combinator to Sam Altman?"
# query_str = "What strategic advice is given about startups?"
# query_str = "Has the author been to Europe?"
# query_str = "What was mentioned about Jessica from pages 17 to 19?"
# query_str = "List all people mentioned in the document."
# query_str = "What experiences from the 1990s are described?"
# query_str = "What programming concepts are given in the document?"
# query_str = "Who are mentioned as colleagues in the document?"
# query_str = "Does the author have any advice on relationships?"
# query_str = "Create table of contents for this article."
# query_str = "How did rejecting prestigious conventional paths lead to the most influential creative projects?"

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

# Enable entity filtering when using entity metadata extraction AND use_entity_filtering is True
enable_entity_filtering = use_entity_filtering and metadata in ["entity", "langextract", "both"]

# Build tools using the factory
keyphrase_tool = rag_factory.get_keyphrase_tool(
    query_str,
    vector_index,
    vector_docstore,
    colbert_reranker,
    llm,
    rag_settings,
    enable_entity_filtering=enable_entity_filtering,
    metadata_option=metadata if metadata else None,
)

page_filter_tool = rag_factory.get_page_filter_tool(
    query_str,
    colbert_reranker,
    vector_index,
    vector_docstore,
    llm,
    metadata_key="source",  # LangExtract uses 'source' for page numbers
    verbose=page_filter_verbose,
)

tools = [
    keyphrase_tool,
    summary_tool,
    page_filter_tool
]

# Build and run the engine
sub_question_engine = rag_factory.build_sub_question_engine(
    tools,
    llm,
    verbose=True
)

print(f"\n📝 ORIGINAL QUERY: {query_str}\n")

# Execute query with retry logic (handled by factory or locally)
response = sub_question_engine.query(query_str)

if response is not None:
    rag_factory.print_response_diagnostics(response)
    print(f"\n📝 RESPONSE:\n{response}\n")

# Cleanup resources
try:
    if 'vector_store' in dir() and hasattr(vector_store, 'client'):
        vector_store.client.release_collection(collection_name=collection_name_vector)
        vector_store.client.close()
except:
    pass

# Suppress warnings and force immediate exit
import warnings
warnings.filterwarnings('ignore')
os._exit(0)
