# import logging
import os
from pathlib import Path
import pprint
from pydantic import Field
from typing import List

# from readable_log_formatter import ReadableFormatter
from llama_index.core import (
                        Settings,
                        StorageContext,
                        VectorStoreIndex,
                        )
from llama_index.core.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
    CBEventType,
)
from llama_index.core.indices.postprocessor import (
                        SentenceTransformerRerank,
                        )
from llama_index.core.node_parser import (
                        SentenceSplitter,
                        )
from llama_index.core.tools import (
                        FunctionTool
                        )
from llama_index.core.vector_stores import MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.readers.file import (
                        PyMuPDFReader,
                        )

import openai
# from trulens_eval import Tru
from utility import (
                get_article_link, 
                get_database_and_sentence_splitter_collection_name,
                get_fusion_accumulate_keyphrase_sort_detail_tool,
                get_fusion_accumulate_page_filter_sort_detail_engine,
                get_fusion_tree_keyphrase_sort_detail_tool,
                get_fusion_tree_page_filter_sort_detail_engine,
                get_page_numbers_from_query_keyphrase,
                get_summary_storage_context,
                get_summary_tree_detail_tool,
                get_vector_store_docstore_and_storage_context,
                )
from database_operation import (
                check_if_milvus_database_collection_exist,
                check_if_mongo_database_namespace_exist
                )

# # Create a logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# # Create a console handler and set the level to DEBUG
# hndl = logging.StreamHandler()
# hndl.setLevel(logging.DEBUG)

# # Use the ReadableFormatter
# hndl.setFormatter(ReadableFormatter())

# # Add the handler to the logger
# logger.addHandler(hndl)


# class CustomLlamaDebugHandler(LlamaDebugHandler):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.logger = logger

#     def _log_event(self, event):
#         # Log the event using the configured logger
#         self.logger.debug(f"Event: {event}")


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
        _documnet, 
        _chunk_size,
        _chunk_overlap
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
    # _nodes = node_parser.get_nodes_from_documents([_documnet])
    _nodes = node_parser.get_nodes_from_documents(_documnet)

    return _nodes


def load_document_nodes_sentence_splitter(
                                _article_link: str,
                                _chunk_size: int,
                                _chunk_overlap: int
                                ) -> List:  
    """
    This function loads a document from a given link, splits it into nodes 
    based on sentence boundaries, and returns these nodes.

    Parameters:
    _article_link (str): The URL or local file path of the document to be loaded.
    _chunk_size (int): The maximum size of each node.
    _chunk_overlap (int): The number of words that should overlap between consecutive 
    nodes.

    Returns:
    _nodes (list): A list of nodes, where each node is a chunk of the document.
    """
    # Only load and parse document if either index or docstore not saved.
    _document = load_document_pdf(_article_link)
    _nodes = get_nodes_from_document_sentence_splitter(
        _document,
        _chunk_size,
        _chunk_overlap
        )
    return _nodes


def create_and_save_vector_index_to_milvus_database(
        _nodes: List,
        _storage_context_vector: StorageContext
        ) -> VectorStoreIndex:
    """
    This function creates and saves a vector index to a Milvus database.

    Parameters:
    _nodes (list): A list of nodes to be indexed. Each node should be an object 
    that can be represented as a vector.
    _storage_context_vector (StorageContext): The storage context for the vector 
    store. This should be an instance of the StorageContext class from the Milvus 
    library.

    Returns:
    VectorStoreIndex: An instance of the VectorStoreIndex class, which represents 
    the created vector index.

    The function takes a list of nodes and a storage context as input, creates 
    a VectorStoreIndex object with these inputs, and then returns this object. 
    The VectorStoreIndex object can be used to perform vector search operations 
    on the indexed nodes.
    """
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    _vector_index = VectorStoreIndex(
        nodes=_nodes,
        storage_context=_storage_context_vector,
        callback_manager=callback_manager,
        )
    return _vector_index


# def get_fusion_accumulate_page_filter_sort_detail_response(
#         query_str_: str = Field(
#             description="A query string that contains instruction on information on specific pages"
#         ), 
#         page_numbers_: List[str] = Field(
#             description="The specific page numbers mentioned iin the query string"
#         )
#         ) -> str:
#     """
#     This function generates a response based on a query string and a list of specific page numbers in
#     this query. It creates a vector retriever with a filter on the specified page numbers, retrieves 
#     relevant nodes, and uses them to generate a response using a fusion accumulate page filter sort 
#     detail engine.

#     Parameters:
#     query_str_ (str): A query string that contains instructions about the information on specific pages.
#     page_numbers_ (List[str]): A list of specific page numbers mentioned in the query string.

#     Returns:
#     str: A response generated based on the query string and the specified page numbers.
#     """
#     # Create a vector retreiver with a filter on page numbers
#     _vector_filter_retriever = vector_index.as_retriever(
#                                     similarity_top_k=similarity_top_k,
#                                     filters=MetadataFilters.from_dicts(
#                                         [{
#                                             "key": "source", 
#                                             "value": page_numbers_,
#                                             "operator": "in"
#                                         }]
#                                     )
#                                 )
    
#     # Calculate the number of nodes retrieved from the vector index on these pages
#     _nodes = _vector_filter_retriever.retrieve(query_str_)

#     _similarity_top_k_filter = len(_nodes)
#     _fusion_top_n_filter = len(_nodes)
#     _num_queries_filter = 1

#     _fusion_accumulate_page_filter_sort_detail_engine = get_fusion_accumulate_page_filter_sort_detail_engine(
#                                                                             _vector_filter_retriever,
#                                                                             _similarity_top_k_filter,
#                                                                             _fusion_top_n_filter,
#                                                                             query_str_,
#                                                                             _num_queries_filter,
#                                                                             )
    
#     _response = _fusion_accumulate_page_filter_sort_detail_engine.query(query_str_)
    
#     return _response


def get_fusion_tree_page_filter_sort_detail_response(
        query_str_: str = Field(
            description="A query string that is intersted only about information on specific pages."
        ), 
        page_numbers_: List[str] = Field(
            description="The specific page numbers mentioned in the query string"
        )
        ) -> str:
    """
    This function generates a response based on a query string and a list of specific page numbers in
    this query. It creates a vector retriever with a filter on the specified page numbers, retrieves 
    relevant nodes, and uses them to generate a response using a fusion tree page filter sort detail 
    engine.

    Parameters:
    query_str_ (str): A query string that contains instructions about the information on specific pages.
    page_numbers_ (List[str]): A list of specific page numbers mentioned in the query string.

    Returns:
    str: A response generated based on the query string and the specified page numbers.
    """
    # Create a vector retreiver with a filter on page numbers
    _vector_filter_retriever = vector_index.as_retriever(
                                    similarity_top_k=similarity_top_k,
                                    filters=MetadataFilters.from_dicts(
                                        [{
                                            "key": "source", 
                                            "value": page_numbers_,
                                            "operator": "in"
                                        }]
                                    )
                                )
    
    # Calculate the number of nodes retrieved from the vector index on these pages
    _nodes = _vector_filter_retriever.retrieve(query_str_)

    _similarity_top_k_filter = len(_nodes)
    _fusion_top_n_filter = len(_nodes)
    _num_queries_filter = 1

    _fusion_tree_page_filter_sort_detail_engine = get_fusion_tree_page_filter_sort_detail_engine(
                                                                            _vector_filter_retriever,
                                                                            _similarity_top_k_filter,
                                                                            _fusion_top_n_filter,
                                                                            query_str_,
                                                                            _num_queries_filter,
                                                                            )
    
    _response = _fusion_tree_page_filter_sort_detail_engine.query(query_str_)
    
    return _response


# Set OpenAI API key, LLM, and embedding model
# openai.api_key = os.environ['OPENAI_API_KEY']
# llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
# Settings.llm = llm

mistral_api_key = os.environ['MISTRAL_API_KEY']
llm = MistralAI(
    model="mistral-large-latest", 
    temperature=0.0,
    max_tokens=2000,
    api_key=mistral_api_key
    )
Settings.llm = llm


# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = embed_model
# embed_model_dim = 384  # for bge-small-en-v1.5
# embed_model_name = "huggingface_embedding_bge_small"

embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
Settings.embed_model = embed_model
embed_model_dim = 1536  # for text-embedding-3-small
embed_model_name = "openai_embedding_3_small"

# Create debug handler
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
Settings.callback_manager = callback_manager

# Create article link
# article_dictory = "metagpt"
# article_name = "metagpt.pdf"

# article_dictory = "uber"
# article_name = "uber_10q_march_2022.pdf"

# article_dictory = "andrew"
# article_name = "eBook-How-to-Build-a-Career-in-AI.pdf"

article_dictory = "paul_graham"
article_name = "paul_graham_essay.pdf"

article_link = get_article_link(article_dictory,
                                article_name
                                )

# Create database and collection names
chunk_method = "sentence_splitter"
chunk_size = 512
chunk_overlap = 128

# chunk_method = "sentence_splitter"
# chunk_size = 256
# chunk_overlap = 50

# Create database name and colleciton names
(database_name, 
collection_name_vector,
collection_name_summary) = get_database_and_sentence_splitter_collection_name(
                                                            article_dictory, 
                                                            chunk_method, 
                                                            embed_model_name, 
                                                            chunk_size,
                                                            chunk_overlap
                                                            )

# Initiate Milvus and MongoDB database
uri_milvus = "http://localhost:19530"
uri_mongo = "mongodb://localhost:27017/"

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

# Create vector store, vector docstore, and vector storage context
(vector_store,
 vector_docstore,
 storage_context_vector) = get_vector_store_docstore_and_storage_context(uri_milvus,
                                                                    uri_mongo,
                                                                    database_name,
                                                                    collection_name_vector,
                                                                    embed_model_dim
                                                                    )

# Create summary summary storage context
storage_context_summary = get_summary_storage_context(uri_mongo,
                                                    database_name,
                                                    collection_name_summary
                                                    )

# for i in list(storage_context.docstore.get_all_ref_doc_info().keys()):
#     print(i)
# print(storage_context.docstore.get_node(leaf_nodes[0].node_id))

# Load documnet nodes if either vector index or docstore not saved.
if save_index_vector or add_document_vector or add_document_summary: 
    extracted_nodes = load_document_nodes_sentence_splitter(
                                                    article_link,
                                                    chunk_size,
                                                    chunk_overlap
                                                    )

if save_index_vector == True:
    # vector_index = create_and_save_vector_index_to_milvus_database(extracted_nodes)
    vector_index = VectorStoreIndex(
        nodes=extracted_nodes,
        storage_context=storage_context_vector,
        callback_manager=callback_manager,
        )

else:
    # Load from Milvus database
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
        )

if add_document_vector == True:
    # Save document nodes to Mongodb docstore at the server
    storage_context_vector.docstore.add_documents(extracted_nodes)

if add_document_summary == True:
    # Save document nodes to Mongodb docstore at the server
    storage_context_summary.docstore.add_documents(extracted_nodes)

# info = storage_context_summary.docstore.get_all_ref_doc_info()

# # Set retriever parameters (with filter)
# similarity_top_k_filter = 6
# num_queries_filter = 1  # for QueryFusionRetriever() in utility.py
# fusion_top_n_filter = 6

# summary_tool: "Useful for summarization or for full context questions related to the documnet"

summary_tool_description = (
            "Useful for summarization or for full context questions related to the document. "
            "Call this function when user ask to: 'Summarize the document' or 'Give me an overview' "
            "or 'What is the main idea of the document?' or 'What is the document about?' "
            "or 'Create document outlines' or 'Create table of contents'."
            )

summary_tool = get_summary_tree_detail_tool(
                                        summary_tool_description,
                                        storage_context_summary
                                        )

# query_str = "What are the keys to building a career in AI?"
# query_str = "What is the importance of networking in AI?"

# query_str = "What is the summary of the MetaGPT paper?"
# query_str = "How do agents share information with other agents?"
# query_str = "What are the high-level results of MetaGPT as described on page 2?"
# query_str = "What are the high-level results of MetaGPT as described on page 1, page 2, and page 3?"
# query_str = "What are the MetaGPT comparisons with ChatDev described on page 8?"
# query_str = "What are the MetaGPT comparisons with ChatDev?"
# query_str = "What are agent roles in MetaGPT, and then how they communicate with each other?"
# query_str = "What are the high-level results of MetaGPT?"
# query_str = "How does metagpt deal with halluciation?"
# query_str = "What is SOPs and how do agents use it?"
# query_str = "Tell me about the ablation study results."
# query_str = (
#     "What could be the potential outcomes of adjusting the amount of safety"
#     " data used in the RLHF stage?"
# )

# query_str = "Tell me the output of the mystery function on 2 and 9."
# query_str = "What is the sum of 2 and 9?"

# query_str = "What are the things that happen in New York?"
# query_str = "Who is Jessica Livingston?"  # NOT A GOOD PROMPT
# query_str = "Who is Jessica?"  # NOT A GOOD PROMPT
# query_str = "What are the things that are mentioned about Sam Altman?"
# query_str = "What are the things that are mentioned about Jessica Livingston?"  # BETTER PROMPT
query_str = "What are the things that are mentioned about Jessica?"  # BETTER PROMPT
# query_str = "Who is Sam Altman?"  # NOT A GOOD PROMPT
# query_str = "Who is Sam?"  # NOT A GOOD PROMPT
# query_str = "What are the things that are mentioned about startups?"
# query_str = "What are mentioned about YC (Y Combinator)?"
# query_str = "What are mentioned about YC (Y Combinator) on pages 19 and 20?"
# query_str = "What are the contents from page 2 to page 4"
# query_str = "What is the summary of the paul graham essay?"
# query_str = "Author's school days."
# query_str = "What are the schools that the author attended?"
# query_str = "What are the specific things that happened at Rhode Island School of Design (RISD)"
# query_str = "What happen in the author's early days?"
# query_str = "What are the specific things that happened in the author's early days?"
# query_str = "Describe the content on pages 19 and 20."
# query_str = "Who have been the president of YC (Y Combinator)?"
# query_str = "What are the thinkgs happened in New York in detail?"
# query_str = "What happened in New York?"
# query_str = "Describe everything that is mentioned about Interleaf one by one."
# query_str = "Describe everything that is mentioned about Interleaf."
# query_str = "Describe everything that is mentioned about Viaweb."
# query_str = "What happened at Interleaf?"
# query_str = "What happened at Interleaf and Viaweb?"
# query_str = "What are the lessions learned by the author from his experience at Interleaf and Viaweb?"
# query_str = (
#     "What are the lessons learned by the author from his experiences at the companies Interleaf"
#      " and Viaweb?")
# query_str = (
#     "What are the lessons learned by the author from his experiences at the companies Interleaf"
#      " and Viaweb? Provide as many details as possible.")
# query_str = (
#     "Give me the main events from page 1 to 4. Provide as many details as possible.")
# query_str = "Give me the main events from page 1 to 4."
# query_str = "What did Paul Graham do in the summer of 1995?"
# query_str = (
#     "What did Paul Graham do in the summer of 1995 and in the couple of "
#     "months afterward?")  # GOOD RESULTS!
# query_str = (
#     "What did Paul Graham do in the summer of 1995 and in the couple of "
#     "months before?")  # THIS PROMPT GOT POOR SCORE (ONLY CHANGE AFTERWARD TO BEFORE)?
# query_str = (
#     "What did Paul Graham do in the summer of 1995 and in the couple of "
#     "months before?")  # THIS PROMPT GOT POOR SCORE (OR EMPTY RESPONSE)?
# query_str = "What did Paul Graham do in the summer of 1995 and earlier in the year?"  # EMPTY RESPONSE!
# query_str = "What did the author do after handing off Y Combinator to Sam Altman?"
query_str = "What did the author's life look like during Y Combinator (YC)?"
# query_str = "What did Paul Graham do in the summer of 1995? Provide as many details as possible."
# query_str = (
#     "What are the specific lessons learned by the author from his experience at the companies Interleaf"
#      " and Viaweb?")
# query_str = "At what school did the author attend a BFA program in painting?"

vector_store.client.load_collection(collection_name=collection_name_vector)

similarity_top_k_keyphrase = 14

# Retrieves page numbers that contain a keyphrase of the query using bm25
page_numbers = get_page_numbers_from_query_keyphrase(
                                                vector_docstore, 
                                                similarity_top_k_keyphrase, 
                                                query_str) 
for p in page_numbers:
    print(f"Page number that contains the keyphrase: {p}")

similarity_top_k = 12
num_queries = 1  # for QueryFusionRetriever() in utility.py
fusion_top_n = 10

# rerank_top_n = 8
rerank_top_n = 8  # for PrevNextNodePostprocessor() with 1 final node

# # Define reranker
# rerank_model = "BAAI/bge-reranker-base"
# rerank = SentenceTransformerRerank(
#     top_n=rerank_top_n,
#     model=rerank_model,
#     )

colbert_reranker = ColbertRerank(
    top_n=rerank_top_n,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)

# # fusion_keyphrase_tool: "Useful for retrieving SPECIFIC context from the document."
# fusion_keyphrase_tool = get_fusion_accumulate_keyphrase_sort_detail_tool(
#                                                                     vector_index,
#                                                                     similarity_top_k,
#                                                                     page_numbers,
#                                                                     fusion_top_n,
#                                                                     query_str,
#                                                                     num_queries,
#                                                                     rerank
#                                                                     )

specific_tool_description = (
            "Useful for retrieving specific, precise, or targeted content from the document. "
            "Use this fuction to pinpoint the relevant information from the document "
            "when user seeks factual answer or a specific detail from the document, "
            "for example, when user uses interrogative words like 'what', 'who', 'where', "
            "'when', 'why', 'how', which may not require understanding the entire document "
            "to provide an answer."
            )

# fusion_keyphrase_tool: "Useful for retrieving SPECIFIC context from the document."
fusion_keyphrase_tool = get_fusion_tree_keyphrase_sort_detail_tool(
                                                            vector_index,
                                                            vector_docstore,
                                                            similarity_top_k,
                                                            page_numbers,
                                                            fusion_top_n,
                                                            query_str,
                                                            num_queries,
                                                            # rerank,
                                                            colbert_reranker,
                                                            specific_tool_description
                                                            )

page_tool_description = (
                "Perform a query search over the page numbers mentioned in the query. "
                "Use this function when you only need to retrieve information from specific pages, "
                "for example when user asks 'What happened on page 19?' "
                "or 'What are the things mentioned on pages 19 and 20?' "
                " or 'Describe the contents from page 1 to page 4'."
                )
fusion_page_filter_tool = FunctionTool.from_defaults(
    name="page_filter_tool",
    fn=get_fusion_tree_page_filter_sort_detail_response,
    description=page_tool_description,
    )

print("\nLLM PREDICT AND CALL:\n\n") 

response = llm.predict_and_call(
                        tools=[
                            fusion_keyphrase_tool,
                            summary_tool,
                            fusion_page_filter_tool
                            ], 
                        user_msg=query_str, 
                        verbose=True
                        )

for i, n in enumerate(response.source_nodes):
    print(f"Item {i+1} of the source pages of response is page: {n.metadata['source']} \
    (with score: {round(n.score, 3) if n.score is not None else None})")
    # print(f"Item {i+1} score: {round(n.score, 4) if n.score is not None else None}\n")
    # print(n)

# # Debug info from callback manager
# # get event time
# print(llama_debug.get_event_time_info(CBEventType.LLM))
# # Get input to LLM and output from LLM information 
# event_pairs = llama_debug.get_llm_inputs_outputs()
# pprint.pprint(event_pairs[1][0])  # human readable format of log
# print(f"\n{event_pairs[1][0].payload['messages'][0].content}") # Input to LLM
# print(f"\n{event_pairs[1][1].payload['response']}")  # Output from LLM


vector_store.client.release_collection(collection_name=collection_name_vector)
vector_store.client.close()  
# Need to do this (otherwise Milvus container will hault when closing)
# del docstore  # MongoDB may frnot need to be manually closed.






