import json
import os
from pathlib import Path

from typing import List, Optional

from llama_index.core import (
                        Settings,
                        StorageContext,
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
from guidance.models import OpenAI as GuidanceOpenAI

from utility_simple import (
                get_article_link, 
                get_database_and_sentence_splitter_collection_name,
                get_fusion_tree_keyphrase_sort_detail_tool_simple,
                get_fusion_tree_page_filter_sort_detail_engine,
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
        device="cpu",
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
                                _metadata: Optional[str] = None
                                ) -> List:  
    """
    This function loads a document from a given link, splits it into nodes based on sentences,
    and optionally extracts entities from the nodes.

    Parameters:
    _article_link (str): The URL of the document to load.
    _chunk_size (int): The maximum size of each node.
    _chunk_overlap (int): The number of words that should overlap between consecutive nodes.
    _metadata (Optional[str], optional): If provided, the function will extract entities from the nodes.
                                        Defaults to None.

    Returns:
    List: A list of nodes, where each node is a dictionary containing the text of the node and optionally,
          the extracted entities.
    """

    # Only load and parse document if either index or docstore not saved.
    _document = load_document_pdf(_article_link)

    if _metadata is not None:
        _nodes = get_nodes_from_document_sentence_splitter_entity_extractor(
            _document,
            _chunk_size,
            _chunk_overlap
            )
    else:
        _nodes = get_nodes_from_document_sentence_splitter(
            _document,
            _chunk_size,
            _chunk_overlap
            )
    return _nodes


def get_fusion_tree_page_filter_sort_detail_tool_simple(
    _query_str: str, 
    # _similarity_top_k_page: int,
    _reranker: ColbertRerank,
    _vector_docstore: MongoDocumentStore,
    ) -> QueryEngineTool:
    
    """
    This function generates a response based on a query string and a list of specific page 
    numbers in this query. It creates a vector retriever with a filter on the specified 
    page numbers, retrieves relevant nodes, and uses them to generate a query engine tool.

    Parameters:
    query_str_ (str): A query string that contains instructions about the information on specific pages.
    page_numbers_ (List[str]): A list of specific page numbers mentioned in the query string.

    Returns:
    str: A response generated based on the query string and the specified page numbers.
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
    print(f"Page_numbers in page filter: {_page_numbers}")

    # Get text nodes from the vector docstore that match the page numbers
    _text_nodes = []
    for node_id, node in _vector_docstore.docs.items():
        if node.metadata['source'] in _page_numbers:
            _text_nodes.append(node) 

    node_length = len(vector_docstore.docs)
    print(f"Node length in docstore: {node_length}")

    # # Create a vector retreiver with a filter on page numbers
    # _vector_retriever = vector_index.as_retriever(
    #                                 similarity_top_k=_similarity_top_k_page,
    #                             )
    # # Retrieve nodes  using the vector retriever and the query
    # scored_nodes = _vector_retriever.retrieve(_query_str)

    # # Extract TextNodes from NodeWithScore objects
    # text_nodes = [scored_node.node for scored_node in scored_nodes]

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
    
    # # Calculate the number of nodes retrieved from the vector index on these pages
    # _nodes = _vector_filter_retriever.retrieve(_query_str)

    _similarity_top_k_filter = len(_text_nodes)
    _fusion_top_n_filter = len(_text_nodes)
    _num_queries_filter = 1

    # _similarity_top_k_filter = _similarity_top_k_page
    # _fusion_top_n_filter = _similarity_top_k_page
    # _num_queries_filter = 1

    # print(f"page filter: {_similarity_top_k_filter}")

    # _similarity_top_k_filter = len(_page_numbers) * 2
    # _fusion_top_n_filter = len(_page_numbers) * 2
    # _num_queries_filter = 1

    _fusion_tree_page_filter_sort_detail_engine = get_fusion_tree_page_filter_sort_detail_engine(
                                                                _vector_filter_retriever,
                                                                _similarity_top_k_filter,
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


# def get_fusion_tree_page_filter_sort_detail_tool(
#     _query_str: str, 
#     _similarity_top_k_page: int,
#     _reranker: ColbertRerank,
#     _vector_docstore: MongoDocumentStore,
#     ) -> QueryEngineTool:
    
#     """
#     This function generates a response based on a query string and a list of specific page 
#     numbers in this query. It creates a vector retriever with a filter on the specified 
#     page numbers, retrieves relevant nodes, and uses them to generate a query engine tool.

#     Parameters:
#     query_str_ (str): A query string that contains instructions about the information on specific pages.
#     page_numbers_ (List[str]): A list of specific page numbers mentioned in the query string.

#     Returns:
#     str: A response generated based on the query string and the specified page numbers.
#     """

#     query_text = (
#     '## Instruction:\n'
#     'Extract all page numbers from the user query. \n'
#     'The page numbers are usually indicated by the phrases "page" or "pages" \n'
#     'Return the page numbers as a list of strings, sorted in ascending order. \n'
#     'Do NOT include "**Output:**" in your response. If no page numbers are mentioned, output ["1"]. \n'

#     '## Examples:\n'
#     '**Query:** "Give me the main events from page 1 to page 4." \n'
#     '**Output:** ["1", "2", "3", "4"] \n'

#     '**Query:** "Give me the main events in the first 6 pages." \n'
#     '**Output:** ["1", "2", "3", "4", "5", "6"] \n'

#     '**Query:** "Summarize pages 10-15 of the document." \n'
#     '**Output:** ["10", "11", "12", "13", "14", "15"] \n'

#     '**Query:** "What are the key findings on page 2?" \n'
#     '**Output:** ["2"] \n'

#     '**Query:** "What is mentioned about YC (Y Combinator) on pages 19 and 20?" \n'
#     '**Output:** ["19", "20"] \n'

#     '**Query:** "What are the lessons learned by the author at the company Interleaf?" \n'
#     '**Output:** ["1"] \n'

#     '## Now extract the page numbers from the following query: \n'

#     '**Query:** {query_str} \n'
#     )

#     # Need to print "1" if no page numbers are mentioned so that this code can run correctly

#     prompt = PromptTemplate(query_text)
#     _page_numbers = llm.predict(prompt=prompt, query_str=_query_str)
#     _page_numbers = json.loads(_page_numbers)  # Convert the string to a list of string
#     print(f"Page_numbers in page filter: {_page_numbers}")

#     # Create a vector retreiver with a filter on page numbers
#     _vector_retriever = vector_index.as_retriever(
#                                     similarity_top_k=_similarity_top_k_page,
#                                 )
#     # Retrieve nodes  using the vector retriever and the query
#     scored_nodes = _vector_retriever.retrieve(_query_str)

#     # Extract TextNodes from NodeWithScore objects
#     text_nodes = [scored_node.node for scored_node in scored_nodes]

#     print(f"Text nodes in page vector index length is: {len(text_nodes)}")
#     for i, n in enumerate(text_nodes):
#         print(f"Item {i+1} of the text nodes in page vector index is page: {n.metadata['source']}")
    

#     _vector_filter_retriever = vector_index.as_retriever(
#                                     similarity_top_k=_similarity_top_k_page,
#                                     filters=MetadataFilters.from_dicts(
#                                         [{
#                                             "key": "source", 
#                                             "value": _page_numbers,
#                                             "operator": "in"
#                                         }]
#                                     )
#                                 )
    
#     # Calculate the number of nodes retrieved from the vector index on these pages
#     _nodes = _vector_filter_retriever.retrieve(_query_str)

#     _similarity_top_k_filter = len(_nodes)
#     _fusion_top_n_filter = len(_nodes)
#     _num_queries_filter = 1

#     # _similarity_top_k_filter = _similarity_top_k_page
#     # _fusion_top_n_filter = _similarity_top_k_page
#     # _num_queries_filter = 1

#     print(f"page filter: {_similarity_top_k_filter}")

#     # _similarity_top_k_filter = len(_page_numbers) * 2
#     # _fusion_top_n_filter = len(_page_numbers) * 2
#     # _num_queries_filter = 1

#     _fusion_tree_page_filter_sort_detail_engine = get_fusion_tree_page_filter_sort_detail_engine(
#                                                                 _vector_filter_retriever,
#                                                                 _similarity_top_k_filter,
#                                                                 _fusion_top_n_filter,
#                                                                 _query_str,
#                                                                 _num_queries_filter,
#                                                                 _reranker,
#                                                                 _vector_docstore,
#                                                                 _page_numbers
#                                                                 )
    
#     _fusion_tree_page_filter_sort_detail_tool = QueryEngineTool.from_defaults(
#                                                         name="page_filter_tool",
#                                                         query_engine=_fusion_tree_page_filter_sort_detail_engine,
#                                                         description=page_tool_description,
#                                                         )

#     return _fusion_tree_page_filter_sort_detail_tool


# nltk.download('punkt_tab')

anthropic_api_key = os.environ['ANTHROPIC_API_KEY']
llm = Anthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.0,
    max_tokens=2000,
    api_key=anthropic_api_key,
    )
Settings.llm = llm

# # Set OpenAI API key and LLM
# openai_api_key = os.environ['OPENAI_API_KEY']
# llm = OpenAI(
#     model="gpt-4o", 
#     temperature=0.0,
#     max_tokens=2000,
#     api_key=openai_api_key
#     )
# Settings.llm = llm

# mistral_api_key = os.environ['MISTRAL_API_KEY']
# llm = MistralAI(
#     model="mistral-large-latest", 
#     temperature=0.0,
#     max_tokens=2000,
#     api_key=mistral_api_key
#     )
# Settings.llm = llm

## Set embedding model
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
metadata = "entity"

# chunk_method = "sentence_splitter"
# chunk_size = 256
# chunk_overlap = 50

# Create database name and colleciton names
# (database_name, 
# collection_name_vector,
# collection_name_summary) = get_database_and_sentence_splitter_collection_name(
#                                                             article_dictory, 
#                                                             chunk_method, 
#                                                             embed_model_name, 
#                                                             chunk_size,
#                                                             chunk_overlap,
#        

# metadata is an optional parameter, will include it if it is not None                                              )
(database_name, 
collection_name_vector,
collection_name_summary) = get_database_and_sentence_splitter_collection_name(
                                                            article_dictory, 
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
# metadata is an optional parameter, will use another function to parse the document
# if the metadata is not None.
if save_index_vector or add_document_vector or add_document_summary: 
    extracted_nodes = load_document_nodes_sentence_splitter(
                                                    article_link,
                                                    chunk_size,
                                                    chunk_overlap,
                                                    metadata
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

# summary_tool_description = (
#             "Useful for summarization or for full context questions related to the document. "
#             "Call this function when user ask to: 'Summarize the document' or 'Give me an overview' "
#             "or 'What is the main idea of the document?' or 'What is the document about?' "
#             "or 'Create document outlines' or 'Create table of contents'."
#             )

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

# query_str = "What are the keys to building a career in AI?"
# query_str = "What is the importance of networking in AI?"

# query_str = "What is the summary of the MetaGPT paper?"
# query_str = "How do agents share information with other agents?"
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
# query_str = "What are the high-level results of MetaGPT as described on page 2?"
# query_str = "What are the high-level results of MetaGPT as described on page 1, page 2, and page 3?"
# query_str = "What are the MetaGPT comparisons with ChatDev described on page 8?"

# query_str = "Tell me the output of the mystery function on 2 and 9."
# query_str = "What is the sum of 2 and 9?"

# query_str = "What are the things that happen in New York?"
# query_str = "Who is Jessica Livingston?"
# query_str = "Who is Jessica?"  # NOT A GOOD PROMPT
# query_str = "What are mentioned about Sam Altman?"
# query_str = "What are the things that are mentioned about startups?"
# query_str = "What are mentioned about YC (Y Combinator)?"
# query_str = "What are mentioned about YC (Y Combinator) on pages 19 and 20?"
# query_str = "What are the contents from page 2 to page 4"
# query_str = "Describe the content on pages 19 and 20."
# query_str = "Describe the content from page 18 to page 21."
# query_str = "Is there anything discussed about Sam Altman from page 18 to page 21?"
# query_str = "What is said about the author on the first 10 pages?"
# query_str = "Describe the content on the first 5 pages?"
# query_str = "What lessons does the author learn on the first 10 pages?"
# query_str = "Give me the main events from page 1 to page 4."
# query_str = "What was mentioned about Jessica on pages 17 and 18?"

query_str = "What was mentioned about Jessica from pages 17 to 22?"

# query_str = "Give me the main events on page 2."
# query_str = "Give me the main events on pages 1 and 2."
# query_str = (
#     "Give me the main events from page 1 to 4. Provide as many details as possible.")
# query_str = "What is the summary of the paul graham essay?"
# query_str = "Author's school days."
# query_str = "What are the schools that the author attended?"
# query_str = "What are the specific things that happened at Rhode Island School of Design (RISD)"
# query_str = "What happen in the author's early days?"
# query_str = "What are the specific things that happened in the author's early days?"
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
#     "What are the lessons learned by the author at the companies Interleaf "
#      "and Viaweb?")
# query_str = (
#     "What are the lessons learned by the author from his experiences at the companies Interleaf"
#      " and Viaweb?")
# query_str = (
#     "What are the lessons learned by the author from his experiences at the companies Interleaf"
#      " and Viaweb? Provide as many details as possible.")
# query_str = (
#     "What was Paul Graham's life at school like and what was it like "
#     "after he handed over YC to Sam Altman?")
# query_str = "What did Paul Graham do in the summer of 1995?"
# query_str = (
#     "What did Paul Graham do in the summer of 1995 and in the couple of "
#     "months after the summer of 1995?")
# query_str = (
#     "What did Paul Graham do in the summer of 1995 and in the couple of "
#     "months afterward?")  # BAD RESULTS!

# query_str = "What did Paul Graham do in 1995 and in 1996?"
# query_str = "What did Paul Graham do in the year 1980, in 1996 and in 2019?"

# query_str = "What did Paul Graham do in 1980, in 1996 and in 2019?"

# query_str = (
#     "What did Paul Graham do in the summer of 1995 and in the couple of "
#     "months before?")  # THIS PROMPT GOT POOR SCORE (ONLY CHANGE AFTERWARD TO BEFORE)?
# query_str = (
#     "What did Paul Graham do in the summer of 1995 and in the couple of "
#     "months before?")  # THIS PROMPT GOT POOR SCORE (OR EMPTY RESPONSE)?
# query_str = "What did Paul Graham do in the summer of 1995 and earlier in the year?"  # EMPTY RESPONSE!
# query_str = "When did the author hand off Y Combinator to Sam Altman?"

# query_str = "What did the author do after handing off Y Combinator to Sam Altman?"

# query_str = "How was the author's life during Y Combinator (YC)?"
# query_str = "When was Y Combinator (YC) founded?"
# query_str = "What did Paul Graham do in the summer of 1995? Provide as many details as possible."
# query_str = (
#     "What are the lessons learned by the author from his experience at the companies Interleaf"
#      " and Viaweb?")
# query_str = "At what school did the author attend a BFA program in painting?"
# query_str = "At what companies did the author work for?"
# query_str = "At what companies did the author work for or as a founder?"
# query_str = "What was the significance of the orange color chosen for Y Combinator's logo?"
# query_str = "Create a table of content for this essay." 

# query_str = "Create table of contents for this article."

# query_str = "What project did the author work on from March 2015 to October 2019?"
# query_str = "What was the author's experience like living in England?"
# query_str = "What was the arrangement between the students and faculty at the Accademia?"


vector_store.client.load_collection(collection_name=collection_name_vector)

# An initial large number making sure nodes of all mentioned pages are retrieved
# This value needs to be larger than the toral number of nodes of the document
# similarity_top_k_page = 60  

# similarity_top_k_keyphrase = 36
# similarity_top_k_fusion = 32
# num_queries = 1  # for QueryFusionRetriever() in utility.py
# fusion_top_n = 28
# rerank_top_n = 20

similarity_top_k_fusion = 36
num_queries = 1  # for QueryFusionRetriever() in utility.py
fusion_top_n = 32
rerank_top_n = 24

# with PrevNextNodePostprocessor() retrieve 8 notes (plus the other note on the same page)

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
keyphrase_tool = get_fusion_tree_keyphrase_sort_detail_tool_simple(
                                                    vector_index,
                                                    vector_docstore,
                                                    similarity_top_k_fusion,
                                                    fusion_top_n,
                                                    query_str,
                                                    num_queries,
                                                    colbert_reranker,
                                                    specific_tool_description
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

                # "Generate multiple sub-questions by using words semantically similar"
                # "towards the original query. "

# Retrieves page numbers that contain a keyphrase of the query using bm25
# page_numbers = get_page_numbers_from_query_keyphrase(
#                                                 vector_docstore, 
#                                                 similarity_top_k_keyphrase, 
#                                                 query_str) 
# for p in page_numbers:
#     print(f"Page number that contains the keyphrase: {p}")

page_filter_tool = get_fusion_tree_page_filter_sort_detail_tool_simple(
                                                    query_str,
                                                    # similarity_top_k_page,
                                                    colbert_reranker,
                                                    vector_docstore,
                                                    )

# fusion_page_filter_tool = FunctionTool.from_defaults(
#     name="page_filter_tool",
#     fn=get_fusion_tree_page_filter_sort_detail_response,
#     description=page_tool_description,
#     )

# print("\nLLM PREDICT AND CALL:\n\n") 

# sub_questions = question_gen.generate(
#                     tools=[
#                         fusion_keyphrase_tool,
#                         summary_tool,
#                         fusion_page_filter_tool
#                         ], 
#                     query=QueryBundle(
#                                 query_str=query_str
#                                 ),
#                 )

# question_gen = GuidanceQuestionGenerator.from_defaults(
#                                 guidance_llm=OpenAI(
#                                             model="gpt-4o",   
#                                             api_key=openai_api_key,
#                                             temperature=0.0,
#                                             ))
# question_gen = GuidanceQuestionGenerator.from_defaults()  # not working
# question_gen = GuidanceQuestionGenerator.from_defaults(guidance_llm=llm)
question_gen = GuidanceQuestionGenerator.from_defaults(
                            guidance_llm=GuidanceOpenAI(
                                model="gpt-4o-2024-11-20",
                                echo=False)
                                )

# tools=[
#     keyphrase_tool,
#     summary_tool,
#     fusion_page_filter_tool
#     ]

# tools=[
#     keyphrase_tool,
#     summary_tool,
#     ]

tools=[
    keyphrase_tool,
    summary_tool,
    page_filter_tool
    ]

sub_question_engine = SubQuestionQueryEngine.from_defaults(
                                        question_gen=question_gen, 
                                        query_engine_tools=tools,
                                        # verbose=False,
                                        verbose=True,
                                        )

# sub_quesrion_tool_description = (
#                 "Perform a search over the document using sub-questions. "
#                 "Use this function when user asks a question that does not "
#                 "mention page number. "
#                 )

# sub_question_tool = QueryEngineTool.from_defaults(
#     name="sub_question_tool",
#     query_engine=sub_question_engine,
#     description=sub_quesrion_tool_description,
#     )

# response = llm.predict_and_call(
#                         tools=[
#                             keyphrase_tool,
#                             summary_tool,
#                             fusion_page_filter_tool
#                             ], 
#                         user_msg=query_str, 
#                         verbose=True
#                         )

try:
    response = sub_question_engine.query(query_str)
except Exception as e:
    print(f"Error getting json answer from LLM: {e}")

for i, n in enumerate(response.source_nodes):
    if bool(n.metadata): # the first few nodes may not have metadata (the LLM response nodes)
        print(f"Item {i+1} of the source pages of response is page: {n.metadata['source']} \
        (with score: {round(n.score, 3) if n.score is not None else None})")
        # print(f"Item {i+1} score: {round(n.score, 4) if n.score is not None else None}\n")
        # print(n)
    else:
        print(f"Item {i+1} question and response:\n{n.text}\n ")

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






