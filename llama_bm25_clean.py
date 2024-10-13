import os
from pathlib import Path
from typing import List

from llama_index.core import (
                        Settings,
                        VectorStoreIndex,
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
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI
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
                get_page_numbers_from_query_keyphrase,
                get_summary_storage_context,
                get_summary_tree_detail_tool,
                get_vector_store_docstore_and_storage_context,
                )
from database_operation import (
                check_if_milvus_database_collection_exist,
                check_if_mongo_database_namespace_exist
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
    _article_link,
    _chunk_size,
    _chunk_overlap
    ) -> List:  
    """
    This function loads a document from a given link, splits it into nodes 
    based on sentence boundaries, and returns these nodes.

    Parameters:
    _article_link (str): The URL or local file path of the document to be loaded.
    _chunk_size (int): The maximum size of each node.
    _chunk_overlap (int): The number of words that should overlap between consecutive nodes.

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
        _nodes,
        _storage_context_vector
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
    _vector_index = VectorStoreIndex(
        nodes=_nodes,
        storage_context=_storage_context_vector,
        )
    return _vector_index


def get_fusion_accumulate_page_filter_sort_detail_response(
        _vector_index: VectorStoreIndex,
        _similarity_top_k: int,
        _query_str: str, 
        _page_numbers: List[str]
        ) -> str:

    # Create a vector retreiver with a filter on page numbers
    _vector_filter_retriever = _vector_index.as_retriever(
                                    similarity_top_k=_similarity_top_k,
                                    filters=MetadataFilters.from_dicts(
                                        [{
                                            "key": "source", 
                                            "value": _page_numbers,
                                            "operator": "in"
                                        }]
                                    )
                                )
    
    # Calculate the number of nodes retrieved from the vector index on these pages
    _nodes = _vector_filter_retriever.retrieve(_query_str)

    _similarity_top_k_filter = len(_nodes)
    _fusion_top_n_filter = len(_nodes)
    _num_queries_filter = 1

    _fusion_accumulate_page_filter_sort_detail_engine = get_fusion_accumulate_page_filter_sort_detail_engine(
                                                                            _vector_filter_retriever,
                                                                            _similarity_top_k_filter,
                                                                            _fusion_top_n_filter,
                                                                            _query_str,
                                                                            _num_queries_filter,
                                                                            )
    
    _response = _fusion_accumulate_page_filter_sort_detail_engine.query(_query_str)
    
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
    vector_index = create_and_save_vector_index_to_milvus_database(extracted_nodes)

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

# Set retriever parameters (no filter)
similarity_top_k = 14
num_queries = 1  # for QueryFusionRetriever() in utility.py
fusion_top_n = 10
rerank_top_n = 6

# similarity_top_k = 15
# num_queries = 1  # for QueryFusionRetriever() in utility.py
# fusion_top_n = 8
# rerank_top_n = 8


# # Set retriever parameters (with filter)
# similarity_top_k_filter = 6
# num_queries_filter = 1  # for QueryFusionRetriever() in utility.py
# fusion_top_n_filter = 6


# (vector_retriever,
# vector_tree_sort_detail_engine) = get_vector_retriever_and_tree_sort_detail_engine(
#                                                                         vector_index,
#                                                                         similarity_top_k,
#                                                                         )

# vector_filter_engine = vector_index.as_query_engine(
#     similarity_top_k=2,
#     filters=MetadataFilters.from_dicts(
#         [
#             {"key": "source", "value": "2"}
#         ]
#     )
# )

# response = vector_filter_engine.query(
#     "What are some high-level results of MetaGPT?", 
# )

# fusion_accumulate_sort_detail_engine = get_fusion_accumulate_sort_detail_engine(
#                                                                         vector_index,
#                                                                         vector_docstore,
#                                                                         similarity_top_k,
#                                                                         num_queries,
#                                                                         fusion_top_n,
#                                                                         )  

# # A fusion tool ("Useful for retrieving specific context from the document.")
# fusion_accumulate_sort_detail_tool = get_fusion_accumulate_sort_detail_tool(
#                                                                     vector_index,
#                                                                     vector_docstore,
#                                                                     similarity_top_k,
#                                                                     num_queries,
#                                                                     fusion_top_n,
#                                                                     )  

# summary_tool: "Useful for summarization questions related to the documnet."
summary_tool = get_summary_tree_detail_tool(
                                        storage_context_summary
                                        )


# The code below does not work (cannot put node_postprocessors here)
# window_retriever = index.as_retriever(
#     similarity_top_k=similarity_top_k,
#     node_postprocessors=[postproc],
#     )  


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

# query_str = "Tell me the output of the mystery function on 2 and 9."
# query_str = "What is the sum of 2 and 9?"

# query_str = "What are the things that happen in New York?"
# query_str = "What are the things that are mentioned about Sam Altman?"
# query_str = "What are the things that are mentioned about startups?"
query_str = "What are mentioned about YC (Y Combinator)?"
# query_str = "What is the summary of the paul graham essay?"
# query_str = "Tell me about his school days."
# query_str = "Tell me about the early days of the author of this essay."
# query_str = "Describe the content on pages 19 and 20."
# query_str = "Who have been the president of YC (Y Combinator)?"
# query_str = "What are the thinkgs happened in New York in detail?"
# query_str = "What happened in New York?"
# query_str = "Describe everything that is mentioned about Interleaf one by one."
# query_str = "Describe everything that is mentioned about Interleaf."
# query_str = "Describe everything that is mentioned about Viaweb."
# query_str = "What happened at Interleaf?"
# query_str = "What happened at Interleaf and Viaweb?"


# query_str = (
#     "What could be the potential outcomes of adjusting the amount of safety"
#     " data used in the RLHF stage?"
# )

# summarize_query_str = "What is the summary of the paul graham essay?"


# Define window post processor
# postproc = MetadataReplacementPostProcessor(
#     target_metadata_key="window"
#     )

vector_store.client.load_collection(collection_name=collection_name_vector)

# # Retrieve nodes, bm25 nodes, and fusion nodes
# nodes = vector_retriever.retrieve(query_str)
# bm25_nodes = bm25_retriever.retrieve(query_str)
# fusion_nodes = fusion_retriever.retrieve(query_str)

# summary_nodes = summary_retriever.retrieve(summarize_query_str)

# Define reranker
rerank_model = "BAAI/bge-reranker-base"
rerank = SentenceTransformerRerank(
    top_n=rerank_top_n,
    model=rerank_model,
    )

# # Get re-ranked fusion nodes
# rerank_nodes = rerank.postprocess_nodes(
#     nodes=fusion_nodes,
#     query_str=query_str,
#     )

# Print retrieved nodes
# print_retreived_nodes("vector", nodes)
# print_retreived_nodes("bm25", bm25_nodes)
# print_retreived_nodes("fusion", fusion_nodes)
# print_retreived_nodes("rerank", rerank_nodes)
# print_retreived_nodes("summary", summary_nodes)


# print(sort_response.get_formatted_sources(length=2000))

# # Create default rerank engine (query_mode="compact")
# rerank_engine = RetrieverQueryEngine.from_args(
#     retriever=fusion_retriever, 
#     node_postprocessors=[rerank],
#     )

# rerank_sort_engine = RetrieverQueryEngine.from_args(
#     retriever=fusion_retriever, 
#     node_postprocessors=[
#         rerank,
#         PageSortNodePostprocessor()
#         ],
#     )

# # Create tree rerank engine (query_mode="compact")
# tree_rerank_engine = RetrieverQueryEngine.from_args(
#     retriever=fusion_retriever, 
#     node_postprocessors=[rerank],
#     response_mode="tree_summarize"
#     )

# tree_rerank_sort_engine = RetrieverQueryEngine.from_args(
#     retriever=fusion_retriever, 
#     node_postprocessors=[
#         rerank,
#         PageSortNodePostprocessor()
#         ],
#     response_mode="tree_summarize"
#     )

# accumulate_rerank_sort_engine = RetrieverQueryEngine.from_args(
#     retriever=fusion_retriever, 
#     node_postprocessors=[
#         rerank,
#         PageSortNodePostprocessor()
#         ],
#     response_mode="accumulate",
#     )

# # Create refine, tree, and accumulate rerank engines 
# (refine_rerank_engine, 
#  tree_rerank_engine, 
#  accumulate_rerank_engine) = get_rerank_refine_tree_and_accumulate_engine_from_retriever(
#                                                                         fusion_retriever,
#                                                                         rerank_top_n,
#                                                                         rerank
#                                                                         )

# summary_prompts_dict = tree_summary_engin.get_prompts()
# type = "tree_summary_engine:"
# display_prompt_dict(type.upper(), summary_prompts_dict)

# # Get default responses
# vector_response = vector_engine.query(query_str)
# bm25_response = bm25_engine.query(query_str)
# fusion_response = fusion_engine.query(query_str)
# rerank_response = rerank_engine.query(query_str)

# # Get default sorted responses
# vector_sort_response = vector_sort_engine.query(query_str)
# bm25_sort_response = bm25_sort_engine.query(query_str)
# fusion_sort_response = fusion_sort_engine.query(query_str)
# rerank_sort_response = rerank_sort_engine.query(query_str)

# Get tree responses
# tree_vector_response = tree_vector_engine.query(query_str)
# tree_bm25_response = tree_bm25_engine.query(query_str)
# tree_fusion_response = tree_fusion_engine.query(query_str)
# tree_rerank_response = tree_rerank_engine.query(query_str)

# Get tree sorted responses
# tree_vector_sort_response = tree_vector_sort_engine.query(query_str)
# tree_bm25_sort_response = tree_bm25_sort_engine.query(query_str)
# tree_fusion_sort_response = tree_fusion_sort_engine.query(query_str)
# tree_rerank_sort_response = tree_rerank_sort_engine.query(query_str)

# Get accumulate sorted responses
# accumulate_fusion_sort_response = accumulate_fusion_sort_engine.query(query_str)
# accumulate_rerank_sort_response = accumulate_rerank_sort_engine.query(query_str)

# tree_summary_engin_response = summary_tree_detail_engine.query(summarize_query_str)

# print(rerank_response.get_formatted_sources(length=2000))



# # Fusion response
# response_fusion = fusion_accumulate_sort_detail_engine.query(
#                                                         query_str
#                                                         )
# print("\nFUSION-ENGINE:\n\n" + str(response_fusion))

# for i, n in enumerate(response_fusion.source_nodes):
#     print(f"Page {i} of fustion response: {n.metadata['source']}\n")


# bm25_retriever = get_bm25_retriever(
#                             vector_docstore,
#                             similarity_top_k,
#                             )

# Use keyphrase filtering
page_numbers = get_page_numbers_from_query_keyphrase(vector_docstore, 
                                                similarity_top_k, 
                                                query_str) 
for p in page_numbers:
    print(f"bm25_nodes_from_keywords page number: {p}")

# fusion_keyphrase_tool: "Useful for retrieving specific context from the document."
fusion_keyphrase_tool = get_fusion_accumulate_keyphrase_sort_detail_tool(
                                                                    vector_index,
                                                                    similarity_top_k,
                                                                    page_numbers,
                                                                    fusion_top_n,
                                                                    query_str,
                                                                    num_queries,
                                                                    )

# page_numbers = ["1","2"]

# # Create vector retreiver (not working by itself as .retrieve())
# vector_filter_retriever = vector_index.as_retriever(
#                                 similarity_top_k=similarity_top_k,
#                                 filters=MetadataFilters.from_dicts(
#                                     [{
#                                         "key": "source", 
#                                         "value": page_numbers,
#                                         "operator": "in"
#                                     }]
#                                 )
#                             )

# nodes = vector_filter_retriever.retrieve(query_str)

# for i, n in enumerate(nodes):
#     print(f"Page {i} of vector_filter_retriever: {n.metadata['source']}\n")

# vector_filter_engine = vector_index.as_query_engine(
#                                 similarity_top_k=similarity_top_k,
#                                 filters=MetadataFilters.from_dicts(
#                                     [{
#                                         "key": "source", 
#                                         "value": page_numbers,
#                                         "operator": "in"
#                                     }]
#                                 )
#                             )

# answer = vector_filter_engine.query(query_str)


# # Get bm25 filter retriever to build a fusion engine with metadata filter (query_str is for getting the nodes first)
# bm25_filter_retriever = get_bm25_filter_retriever(vector_filter_retriever, 
#                                             query_str, 
#                                             similarity_top_k
#                                             )

# # Get fusion accumulate filter sort detail engine
# fusion_accumulate_filter_sort_detail_engine = get_fusion_accumulate_filter_sort_detail_engine(
#                                                                     vector_filter_retriever,
#                                                                     bm25_filter_retriever,
#                                                                     fusion_top_n,
#                                                                     num_queries
#                                                                     )



# fusion_accumulate_keyphrase_sort_detail_tool = QueryEngineTool.from_defaults(
#     name="fusion_keyphrase_tool",
#     query_engine=fusion_accumulate_filter_sort_detail_engine,
#     description=(
#         "Useful for retrieving specific context from the document."
#     ),
# )


# # Fusion response with metadata filter (from limited pages)
# response_fusion_filter = fusion_accumulate_filter_sort_detail_engine.query(
#                                                         query_str
#                                                         )
# print("\nFUSION-KEYPHRASE-ENGINE:\n\n" + str(response_fusion_filter))

# for i, n in enumerate(response_fusion_filter.source_nodes):
#     print(f"Page {i} of fustion keyphrase response: {n.metadata['source']}\n")





# # Vector response with metadata filter (from limited pages)
# response_vector_filter = vector_tree_filter_sort_detail_engine.query(
#                                                         query_str
#                                                         )
# print("\nVECTOR-FILTER-ENGINE:\n\n" + str(response_vector_filter))
# for n in response_vector_filter.source_nodes:
#     print(f"vector filter page number: {n.metadata['source']}")


# # Add function tools
# add_tool = FunctionTool.from_defaults(fn=add)
# mystery_tool = FunctionTool.from_defaults(fn=mystery)

fusion_page_filter_tool = FunctionTool.from_defaults(
    name="page_filter_tool",
    fn=get_fusion_accumulate_page_filter_sort_detail_response
)


# # Demonstrate function tool
# response_tool = llm.predict_and_call(
#                             [add_tool, mystery_tool], 
#                             "Tell me the output of the mystery function on 2 and 9", 
#                             verbose=True
#                             )
# print(f"\n{response_tool}\n\n")


# Define query engine tool with page numbers filter


# router_query_engine = RouterQueryEngine(
#     selector=LLMSingleSelector.from_defaults(),
#     query_engine_tools=[
#         summary_tree_detail_tool,
#         fusion_accumulate_sort_detail_tool,
#     ],
#     verbose=True
# )

# response = router_query_engine.query(query_str)

# print("\nROUTER-TOOL-ENGINE (ALL PAGES):\n\n" + str(response))

# response = llm.predict_and_call(
#                         [summary_tree_detail_tool, 
#                          mystery_tool], 
#                         query_str, 
#                         verbose=True
#                         )

# response = llm.predict_and_call(
#                         tools=[fusion_accumulate_filter_sort_detail_tool], 
#                         user_msg=query_str, 
#                         verbose=True
#                         )

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

# response = llm.predict_and_call(
#                         tools=[
#                             fusion_keyphrase_tool,
#                             summary_tool,
#                             fusion_page_filter_tool,
#                             add_tool,
#                             mystery_tool
#                             ], 
#                         user_msg=query_str, 
#                         verbose=True
#                         )

# print("\nLLM PREDICT AND CALL:\n\n" + str(response))

for i, n in enumerate(response.source_nodes):
    print(f"Item {i+1} of the source page of response is page: {n.metadata['source']}\n")

# Prints responses
# print("\nVECTOR-ENGINE:\n\n" + str(vector_response))
# print("\nVECTOR-SORT-ENGINE:\n\n" + str(vector_sort_response))
# print("\nTREE-VECTOR-ENGINE:\n\n" + str(tree_vector_response))
# print("\nTREE-VECTOR-SORT-ENGINE:\n\n" + str(tree_vector_sort_response))

# print("\nBM25-ENGINE:\n\n" + str(bm25_response))
# print("\nBM25-SORT-ENGINE:\n\n" + str(bm25_sort_response))
# print("\nTREE-BM25-ENGINE:\n\n" + str(tree_bm25_response))
# print("\nTREE-BM25-SORT-ENGINE:\n\n" + str(tree_bm25_sort_response))

# print("\nFUSION-ENGINE:\n\n" + str(fusion_response))
# print("\nFUSION-SORT-ENGINE:\n\n" + str(fusion_sort_response))
# print("\nTREE-FUSION-ENGINE:\n\n" + str(tree_fusion_response))
# print("\nTREE-FUSION-SORT-ENGINE:\n\n" + str(tree_fusion_sort_response))

# print("\nRERANK-ENGINE:\n\n" + str(rerank_response))
# print("\nRERANK-SORT-ENGINE:\n\n" + str(rerank_sort_response))
# print("\nTREE-RERANK-ENGINE:\n\n" + str(tree_rerank_response))
# print("\nTREE-RERANK-SORT-ENGINE:\n\n" + str(tree_rerank_sort_response))

# print("\nACCUMULATE-FUSION-SORT-ENGINE:\n\n" + str(accumulate_fusion_sort_response))
# print("\nACCUMULATE-RERANK-SORT-ENGINE:\n\n" + str(accumulate_rerank_sort_response))

# print("\nTREE-SUMMARY-ENGINE:\n\n" + str(tree_summary_engin_response))


vector_store.client.release_collection(collection_name=collection_name_vector)
vector_store.client.close()  
# Need to do this (otherwise Milvus container will hault when closing)
# del docstore  # MongoDB may frnot need to be manually closed.






