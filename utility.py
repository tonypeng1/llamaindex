from typing import List, Optional

from keybert import KeyBERT
from llama_index.core import (
                        PromptTemplate,
                        QueryBundle,
                        StorageContext,
                        SummaryIndex,
                        VectorStoreIndex
                        )
from llama_index.core.indices.postprocessor import (
                        AutoPrevNextNodePostprocessor,
                        PrevNextNodePostprocessor,
                        SentenceTransformerRerank,
                        SimilarityPostprocessor,
                        )
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.vector_stores.milvus import MilvusVectorStore



class SortNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
            self, 
            nodes: List[NodeWithScore], 
            query_bundle: Optional[QueryBundle]
            ) -> List[NodeWithScore]:
        
        # Custom post-processor: Order nodes based on the order it appears in a document (using "start_char_idx")

        # Create new node dictionary
        _nodes_dic = [{"start_char_idx": node.node.start_char_idx, "node": node} for node in nodes]

        # Sort based on start_char_idx
        sorted_nodes_dic = sorted(_nodes_dic, key=lambda x: x["start_char_idx"])

        # Get the new nodes from the sorted node dic
        sorted_new_nodes = [node["node"] for node in sorted_nodes_dic]

        return sorted_new_nodes


class PageSortNodePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle]
            ) -> List[NodeWithScore]:
        
        # Custom post-processor: Order nodes first based on page number and then based on
        # the order it appears in a document (using "start_char_idx")

        # Create new node dictionary
        _nodes_dic = [{"source": node.node.metadata["source"], \
                       "start_char_idx": node.node.start_char_idx, \
                        "node": node} for node in nodes]

        # Sort based on page_label and then start_char_idx
        sorted_nodes_dic = sorted(_nodes_dic, \
                                    key=lambda x: (int(x["source"]), x["start_char_idx"]))

        # Get the new nodes from the sorted node dic
        sorted_new_nodes = [node["node"] for node in sorted_nodes_dic]

        return sorted_new_nodes


def get_article_link(
        _article_dictory, 
        _article_name
        ):
    
    return "./data/" + _article_dictory + "/" + _article_name


def get_database_and_window_collection_name(
        _article_dictory, 
        _chunk_method, 
        _embed_model_name,
        _window_size
        ):
    
    _database_name = _article_dictory + "_" + _chunk_method
    _collection_name = _embed_model_name + "_window_size_" + str(_window_size)
    return _database_name, _collection_name


def get_database_and_sentence_splitter_collection_name(
        _article_dictory, 
        _chunk_method, 
        _embed_model_name,
        _chunk_size,
        _chunk_overlap,
        ):
    
    _database_name = _article_dictory + "_" + _chunk_method
    _collection_name = _embed_model_name \
                            + "_chunk_size_" + str(_chunk_size) \
                            + "_chunk_overlap_" + str(_chunk_overlap)
    _collection_name_summary = _embed_model_name \
                            + "_chunk_size_" + str(_chunk_size) \
                            + "_chunk_overlap_" + str(_chunk_overlap) \
                            + "_summary"

    return _database_name, _collection_name, _collection_name_summary


def get_database_and_llamaparse_collection_name(
        _article_dictory, 
        _chunk_method, 
        _embed_model_name,
        _parse_method,
        ):
    
    _database_name = _article_dictory + "_" + _chunk_method
    _collection_name = _embed_model_name \
                            + "_parse_method_" + _parse_method
    _collection_name_summary = _embed_model_name \
                            + "_parse_method_" + _parse_method \
                            + "_summary"

    return _database_name, _collection_name, _collection_name_summary


def get_database_and_automerge_collection_name(
        _article_dictory, 
        _chunk_method, 
        _embed_model_name,
        chunk_sizes,
        ):
    
    _database_name = _article_dictory + "_" + _chunk_method
    _collection_name = _embed_model_name + "_size_" + str(chunk_sizes[0]) + "_" + \
                        str(chunk_sizes[1]) + "_" + str(chunk_sizes[2])
    
    return _database_name, _collection_name


def get_compact_tree_and_accumulate_engine_from_index(
        _index,
        _similarity_top_k,
        _postproc
        ):
    
    _compact_engine = _index.as_query_engine(
                                similarity_top_k=_similarity_top_k,
                                node_postprocessors=[_postproc],
                                response_mode="compact",
                                )
    _tree_engine = _index.as_query_engine(
                                similarity_top_k=_similarity_top_k,
                                node_postprocessors=[_postproc],
                                response_mode="tree_summarize",
                                )
    _accumulate_engine = _index.as_query_engine(
                                similarity_top_k=_similarity_top_k,
                                node_postprocessors=[_postproc],
                                response_mode="accumulate",
                                )
    return _compact_engine, _tree_engine, _accumulate_engine


def get_rerank_compact_tree_and_accumulate_engine_from_index(
        _index,
        _similarity_top_k,
        _postproc,
        _rerank
        ):
    
    _compact_rerank_engine = _index.as_query_engine(
                            similarity_top_k=_similarity_top_k,
                            node_postprocessors=[_postproc, _rerank],
                            response_mode="compact",
                            )

    _tree_rerank_engine = _index.as_query_engine(
                                similarity_top_k=_similarity_top_k,
                                node_postprocessors=[_postproc, _rerank],
                                response_mode="tree_summarize",
                                )

    _accumulate_rerank_engine = _index.as_query_engine(
                                similarity_top_k=_similarity_top_k,
                                node_postprocessors=[_postproc, _rerank],
                                response_mode="accumulate",
                                )
    return _compact_rerank_engine, _tree_rerank_engine, _accumulate_rerank_engine


def get_default_query_engine_from_retriever(
    retriever_1,
    retriever_2,
    ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1
        )
    query_engine_2 = RetrieverQueryEngine.from_args(
        retriever=retriever_2
        )

    return query_engine_1, query_engine_2


def get_tree_query_engine_from_retriever(
        retriever_1,
        retriever_2,
        ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1, 
        response_mode="tree_summarize",
        )

    query_engine_2 = RetrieverQueryEngine.from_args(
        retriever=retriever_2, 
        response_mode="tree_summarize",
        )

    return query_engine_1, query_engine_2


# def get_tree_query_engine_with_sort_from_retriever(
#         retriever_1,
#         retriever_2,
#         ):

#     query_engine_1 = RetrieverQueryEngine.from_args(
#         retriever=retriever_1, 
#         response_mode="tree_summarize",
#         node_postprocessors=[SortNodePostprocessor()],
#         )

#     query_engine_2 = RetrieverQueryEngine.from_args(
#         retriever=retriever_2, 
#         response_mode="tree_summarize",
#         node_postprocessors=[SortNodePostprocessor()],
#         )

#     return query_engine_1, query_engine_2


def get_tree_engine_from_retriever(
        retriever_1,
        retriever_2,
        retriever_3,
        retriever_4,
        ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1, 
        response_mode="tree_summarize",
        )

    query_engine_2 = RetrieverQueryEngine.from_args(
        retriever=retriever_2, 
        response_mode="tree_summarize",
        )

    query_engine_3 = RetrieverQueryEngine.from_args(
        retriever=retriever_3, 
        response_mode="tree_summarize",
        )
    
    query_engine_4 = RetrieverQueryEngine.from_args(
        retriever=retriever_4, 
        response_mode="tree_summarize",
        use_async= True,
        )

    return query_engine_1, query_engine_2, query_engine_3, query_engine_4


# def get_tree_engine_with_sort_from_retriever(
#         retriever_1,
#         retriever_2,
#         retriever_3,
#         ):

#     query_engine_1 = RetrieverQueryEngine.from_args(
#         retriever=retriever_1, 
#         response_mode="tree_summarize",
#         node_postprocessors=[SortNodePostprocessor()],
#         )

#     query_engine_2 = RetrieverQueryEngine.from_args(
#         retriever=retriever_2, 
#         response_mode="tree_summarize",
#         node_postprocessors=[SortNodePostprocessor()],
#         )

#     query_engine_3 = RetrieverQueryEngine.from_args(
#         retriever=retriever_3, 
#         response_mode="tree_summarize",
#         node_postprocessors=[SortNodePostprocessor()],
#         )

#     return query_engine_1, query_engine_2, query_engine_3


def get_accumulate_query_engine_from_retriever(
        retriever_1,
        retriever_2,
        ):

    query_engine_1 = RetrieverQueryEngine.from_args(
        retriever=retriever_1, 
        response_mode="accumulate",
        )

    retriever_2_engine = RetrieverQueryEngine.from_args(
        retriever=retriever_2, 
        response_mode="accumulate",
        )

    return query_engine_1, retriever_2_engine


# def get_accumulate_query_engine_with_sort_from_retriever(
#         retriever_1,
#         retriever_2,
#         ):

#     query_engine_1 = RetrieverQueryEngine.from_args(
#         retriever=retriever_1, 
#         response_mode="accumulate",
#         node_postprocessors=[SortNodePostprocessor()],
#         )

#     retriever_2_engine = RetrieverQueryEngine.from_args(
#         retriever=retriever_2, 
#         response_mode="accumulate",
#         node_postprocessors=[SortNodePostprocessor()],
#         )

#     return query_engine_1, retriever_2_engine


def get_vector_store_docstore_and_storage_context(_uri_milvus,
                                                _uri_mongo,
                                                _database_name,
                                                _collection_name_vector,
                                                _embed_model_dim):
    
    # Initiate vector store (a new empty collection will be created in Milvus server)
    _vector_store = MilvusVectorStore(
        uri=_uri_milvus,
        db_name=_database_name,
        collection_name=_collection_name_vector,
        dim=_embed_model_dim,  # dim of HuggingFace "BAAI/bge-small-en-v1.5" embedding model
        enable_dynamic_field=False,
        )

    # Initiate MongoDB vector docstore (Not yet save to MongoDB server)
    _vector_docstore = MongoDocumentStore.from_uri(
        uri=_uri_mongo,
        db_name=_database_name,
        namespace=_collection_name_vector
        )

    # Initiate vector storage context: use Milvus as vector store and Mongo as docstore 
    _storage_context_vector = StorageContext.from_defaults(
        vector_store=_vector_store,
        docstore=_vector_docstore
        )
    
    return _vector_store, _vector_docstore, _storage_context_vector


def get_llamaparse_vector_store_docstore_and_storage_context(_uri_milvus,
                                                _uri_mongo,
                                                _database_name,
                                                _collection_name_vector,
                                                _embed_model_dim):
    
    # Initiate vector store (a new empty collection will be created in Milvus server)
    _vector_store = MilvusVectorStore(
        uri=_uri_milvus,
        db_name=_database_name,
        collection_name=_collection_name_vector,
        dim=_embed_model_dim,  # dim of HuggingFace "BAAI/bge-small-en-v1.5" embedding model
        enable_dynamic_field=False,
        )

    # Initiate MongoDB vector docstore (Not yet save to MongoDB server)
    _vector_docstore = MongoDocumentStore.from_uri(
        uri=_uri_mongo,
        db_name=_database_name,
        namespace=_collection_name_vector
        )

    # Initiate vector storage context: use Milvus as vector store and Mongo as docstore 
    _storage_context_vector = StorageContext.from_defaults(
        vector_store=_vector_store,
        docstore=_vector_docstore
        )
    
    return _vector_store, _vector_docstore, _storage_context_vector


def get_summary_storage_context(_uri_mongo,
                                _database_name,
                                _collection_name_summary):

    # Initiate MongoDB summary docstore (Not yet save to MongoDB server)
    _summary_docstore = MongoDocumentStore.from_uri(
        uri=_uri_mongo,
        db_name=_database_name,
        namespace=_collection_name_summary
        )

    # Initiate summary storage context: use Milvus as vector store and Mongo as docstore 
    _storage_context_summary = StorageContext.from_defaults(
        docstore=_summary_docstore
        )
    
    return _storage_context_summary


def get_summary_tree_detail_engine(_storage_context_summary):

    # Get nodes from summary storage context
    _extracted_nodes = list(_storage_context_summary.docstore.docs.values())
    # Create index
    _summary_index = SummaryIndex(
                        nodes=_extracted_nodes
                        )
    # Create retriever aka query engine
    _summary_retriever = _summary_index.as_retriever()
    _tree_summary_engine = RetrieverQueryEngine.from_args(
                                        retriever=_summary_retriever, 
                                        response_mode="tree_summarize",
                                        # node_postprocessors=[PageSortNodePostprocessor()],
                                        use_async= True,
                                        )
    _tree_summary_detail_engine = change_summary_engine_prompt_to_in_detail(_tree_summary_engine)

    return _tree_summary_detail_engine


def get_summary_retriever_and_tree_detail_engine(_storage_context_summary):

    # Get nodes from summary storage context
    _extracted_nodes = list(_storage_context_summary.docstore.docs.values())
    # Create index
    _summary_index = SummaryIndex(
                        nodes=_extracted_nodes
                        )
    # Create retriever aka query engine
    _summary_retriever = _summary_index.as_retriever()
    _tree_summary_engine = RetrieverQueryEngine.from_args(
                                        retriever=_summary_retriever, 
                                        response_mode="tree_summarize",
                                        use_async= True,
                                        )
    _tree_summary_detail_engine = change_summary_engine_prompt_to_in_detail(_tree_summary_engine)

    return _summary_retriever, _tree_summary_detail_engine


def get_vector_retriever_and_tree_sort_detail_engine(
        _index,
        _similarity_top_k,
        ):

    _retriever = _index.as_retriever(
        similarity_top_k=_similarity_top_k
        )
    _tree_sort_engine = RetrieverQueryEngine.from_args(
        retriever=_retriever, 
        response_mode="tree_summarize",
        node_postprocessors=[PageSortNodePostprocessor()],
        )
    _tree_sort_detail_engine = change_tree_engine_prompt_to_in_detail(_tree_sort_engine)

    return _retriever, _tree_sort_detail_engine


def get_bm25_retriever(
        _vector_docstore,
        _similarity_top_k,
        ):
    
    _retriever = BM25Retriever.from_defaults(
        similarity_top_k=_similarity_top_k,
        docstore=_vector_docstore,
        )

    return _retriever


def get_bm25_retriever_and_tree_sort_detail_engine(
        _vector_docstore,
        _similarity_top_k,
        ):

    _retriever = BM25Retriever.from_defaults(
        similarity_top_k=_similarity_top_k,
        docstore=_vector_docstore,
        )
    _tree_sort_engine = RetrieverQueryEngine.from_args(
        retriever=_retriever, 
        response_mode="tree_summarize",
        node_postprocessors=[PageSortNodePostprocessor()],
        )
    _tree_sort_detail_engine = change_tree_engine_prompt_to_in_detail(_tree_sort_engine)

    return _retriever, _tree_sort_detail_engine


def get_fusion_retriever(
        _vector_index,
        _vector_docstore,
        _similarity_top_k,
        _num_queries,
        _fusion_top_n,
        ):
    
    _vector_retriever = _vector_index.as_retriever(
        similarity_top_k=_similarity_top_k
        )
    _bm25_retriever = BM25Retriever.from_defaults(
        similarity_top_k=_similarity_top_k,
        docstore=_vector_docstore,
        )
    _retriever = QueryFusionRetriever(
        retrievers=[_vector_retriever, _bm25_retriever],
        similarity_top_k=_fusion_top_n,
        num_queries=_num_queries,  # set this to 1 to disable query generation
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
        # query_gen_prompt="...",  # for overriding the query generation prompt
        )
    
    return _retriever


def get_fusion_retriever_and_tree_sort_detail_engine(
        _vector_index,
        _vector_docstore,
        _similarity_top_k,
        _num_queries,
        _fusion_top_n,
        ):

    _vector_retriever = _vector_index.as_retriever(
        similarity_top_k=_similarity_top_k
        )
    _bm25_retriever = BM25Retriever.from_defaults(
        similarity_top_k=_similarity_top_k,
        docstore=_vector_docstore,
        )
    _retriever = QueryFusionRetriever(
        retrievers=[_vector_retriever, _bm25_retriever],
        similarity_top_k=_fusion_top_n,
        num_queries=_num_queries,  # set this to 1 to disable query generation
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
        # query_gen_prompt="...",  # for overriding the query generation prompt
        )
    _tree_sort_engine = RetrieverQueryEngine.from_args(
        retriever=_retriever, 
        response_mode="tree_summarize",
        node_postprocessors=[PageSortNodePostprocessor()],
        )
    
    _tree_sort_detail_engine = change_tree_engine_prompt_to_in_detail(_tree_sort_engine)

    return _retriever, _tree_sort_detail_engine


def get_keyphrase_from_query_str(query_str) -> str:
    """
    Extracts a single keyphrase from a given query string using the KeyBERT model.

    The keyphrase is a sequence of one to three words that best represents the main
    topic of the query string. The function uses the KeyBERT model to perform this
    extraction.

    Parameters:
    query_str (str): The input query string from which to extract the keyphrase.

    Returns:
    str: A single keyphrase string that best represents the main topic of the query string.
    """

    # Initialize KeyBERT model
    kw_model = KeyBERT()

    # Extract keywords (keyphrases) from the query string
    keywords = kw_model.extract_keywords(
                                docs=query_str, 
                                keyphrase_ngram_range=(3, 3), 
                                # use_maxsum=True,
                                # nr_candidates=5,
                                top_n=3,
                                )
    keywords_set = set()
    for k in keywords:
        keywords_set.update(k[0].split())

    final_keywords = str()
    for i in keywords_set:
        final_keywords = final_keywords + " " + i

    return final_keywords.lstrip()  # Remove leading whitespace and return the final keyphrase



def get_page_numbers_using_bm25_and_keyphrase(_bm25_retriever, _keyword) -> list:
    """
    This function retrieves a list of page numbers of the nodes that contain a given keyphrase in a 
    document using BM25 retriever. The page numbers are sorted in ascending order.

    Parameters:
    _bm25_retriever (BM25Retriever): An instance of the BM25Retriever class.
    _keyword (str): The keyword to search for in the documents.

    Returns:
    list: A list of page numbers (as strings) that contain the keyphrase, sorted in ascending order.
    """

    bm25_score_nodes = _bm25_retriever.retrieve(_keyword)
    bm25_text_nodes = [node.node for node in bm25_score_nodes ]  # get TextNode from ScoredNode

    # page_numbers = sorted([int(n.metadata["source"]) for n in bm25_text_nodes])  # get and sort the page numbers
    page_numbers = [int(n.metadata["source"]) for n in bm25_text_nodes]  # get and sort the page numbers
    page_numbers = [str(p) for p in page_numbers]  # convert back to str

    return page_numbers



def get_vector_tree_filter_sort_detail_engine(_vector_filter_retriever):

    _vector_tree_filter_sort_engine = RetrieverQueryEngine.from_args(
                                                retriever=_vector_filter_retriever, 
                                                response_mode="tree_summarize",
                                                node_postprocessors=[PageSortNodePostprocessor()],
                                                )

    _vector_tree_filter_sort_detail_engine = change_tree_engine_prompt_to_in_detail(
                                                            _vector_tree_filter_sort_engine
                                                            )
    
    return _vector_tree_filter_sort_detail_engine


def get_bm25_filter_retriever(_vector_filter_retriever, 
                              _query_str: str, 
                              _similarity_top_k: int)-> BM25Retriever:
    """
    This function creates a BM25 filter retriever using the nodes retrieved from a vector 
    filter retriever and a query string.

    Parameters:
    _vector_filter_retriever (BaseRetriever): An instance of a vector retriever class that 
                                        supports retrieving nodes based on a query string.
    _query_str (str): The query string used to retrieve nodes from the vector filter retriever.
    _similarity_top_k (int): The number of top similar nodes to consider for the BM25 filter 
                            retriever.

    Returns:
    _bm25_filter_retriever (BM25Retriever): An instance of a BM25Retriever class that uses the retrieved filtered nodes 
    from the vector filter retriever on a query string.
    """

    # Get nodes for bm25 filter retriever
    score_nodes = _vector_filter_retriever.retrieve(_query_str)
    text_nodes = [node.node for node in score_nodes ]  # get TextNode from ScoredNode

    # Create bm25 filter engine using filtered nodes
    _bm25_filter_retriever = BM25Retriever.from_defaults(
                                similarity_top_k=_similarity_top_k,
                                nodes=text_nodes,
                                )

    return _bm25_filter_retriever


def get_fusion_accumulate_filter_sort_detail_engine(_vector_retriever,
                                                    _bm25_filter_retriever: BM25Retriever,
                                                    _fusion_top_n: int,
                                                    _num_queries: int,
                                                    _rerank: SentenceTransformerRerank,
                                                    ):
    """
    This function creates a fusion filter retriever and engine that combines the results of two 
    retrievers: a vector filter retriever and a BM25 filter retriever. The results are then sorted 
    and reranked using a SentenceTransformerRerank object. The engine is configured to accumulate 
    results and provide a detailed response.

    Args:
    _vector_filter_retriever: The first retriever to be used in the fusion.
    _bm25_filter_retriever (BM25Retriever): The second retriever to be used in the fusion.
    _fusion_top_n (int): The number of top results to consider from each retriever.
    _num_queries (int): The number of queries to generate. Set this to 1 to disable query generation.
    _rerank (SentenceTransformerRerank): The reranking model to be used.

    Returns:
    RetrieverQueryEngine: A query engine that uses the fusion filter retriever and the specified reranking model.
    The engine is configured to accumulate results and provide a detailed response.
    """    
    # Create fusion filter retreiver and engine
    _fusion_filter_retriever = QueryFusionRetriever(
                                retrievers=[
                                        _vector_retriever, 
                                        _bm25_filter_retriever
                                        ],
                                similarity_top_k=_fusion_top_n,
                                num_queries=_num_queries,  # set this to 1 to disable query generation
                                mode="reciprocal_rerank",
                                retriever_weights=[0.5, 0.5],
                                use_async=True,
                                verbose=True,
                                # query_gen_prompt="...",  # for overriding the query generation prompt
                                )

    _fusion_accumulate_filter_sort_engine = RetrieverQueryEngine.from_args(
                                            retriever=_fusion_filter_retriever, 
                                            node_postprocessors=[
                                                _rerank,
                                                PageSortNodePostprocessor(),
                                                ],
                                            response_mode="accumulate",
                                            )

    _fusion_accumulate_filter_sort_detail_engine = change_accumulate_engine_prompt_to_in_detail(
                                                            _fusion_accumulate_filter_sort_engine
                                                            )
    
    return _fusion_accumulate_filter_sort_detail_engine


def get_fusion_tree_filter_sort_detail_engine(_vector_retriever,
                                            _vector_docstore: MongoDocumentStore,
                                            _bm25_filter_retriever: BM25Retriever,
                                            _fusion_top_n: int,
                                            _num_queries: int,
                                            # _rerank: SentenceTransformerRerank = None,
                                            _rerank: ColbertRerank = None,
                                            ):
    """
    This function creates a fusion filter retriever and engine that combines the results of two 
    retrievers: a vector filter retriever and a BM25 filter retriever. The results are then sorted 
    and reranked using a SentenceTransformerRerank object. The engine is configured to tree 
    results with a similarity cutoff of 0.7 and provide a detailed response.

    Args:
    _vector_filter_retriever: The first retriever to be used in the fusion.
    _bm25_filter_retriever (BM25Retriever): The second retriever to be used in the fusion.
    _fusion_top_n (int): The number of top results to consider from each retriever.
    _num_queries (int): The number of queries to generate. Set this to 1 to disable query generation.
    _rerank (SentenceTransformerRerank): The reranking model to be used.

    Returns:
    RetrieverQueryEngine: A query engine that uses the fusion filter retriever and the specified reranking model.
    The engine is configured to tree results and provide a detailed response.
    """    
    # Create fusion filter retreiver and engine
    _fusion_filter_retriever = QueryFusionRetriever(
                                retrievers=[
                                        _vector_retriever, 
                                        _bm25_filter_retriever
                                        ],
                                similarity_top_k=_fusion_top_n,
                                # similarity_top_k=2,
                                num_queries=_num_queries,  # set this to 1 to disable query generation
                                mode="reciprocal_rerank",
                                retriever_weights=[0.5, 0.5],
                                use_async=True,
                                verbose=True,
                                # query_gen_prompt="...",  # for overriding the query generation prompt
                                )

    AutoPrevNext = AutoPrevNextNodePostprocessor(
                                        docstore=_vector_docstore,
                                        num_nodes=5,
                                        verbose=True,
                                        )

    PrevNext = PrevNextNodePostprocessor(
                                    docstore=_vector_docstore,
                                    num_nodes=2,  # each page now has two nodes, one with next, the other previous
                                    mode="both",
                                    verbose=True,
                                    )  # can only retrieve the two nodes on one page 

    node_postprocessors = [
                        # SimilarityPostprocessor(similarity_cutoff=0.7),
                        PrevNext,
                        # AutoPrevNext,
                        PageSortNodePostprocessor(),
                        ]
    
    if _rerank is not None:
        node_postprocessors.insert(0, _rerank)

    _fusion_tree_filter_sort_engine = RetrieverQueryEngine.from_args(
                                            retriever=_fusion_filter_retriever, 
                                            node_postprocessors=node_postprocessors,
                                            response_mode="tree_summarize",
                                            )

    _fusion_tree_filter_sort_detail_engine = change_tree_engine_prompt_to_in_detail(
                                                            _fusion_tree_filter_sort_engine
                                                            )
    
    return _fusion_tree_filter_sort_detail_engine


def get_page_numbers_from_query_keyphrase(_vector_docstore, _similarity_top_k, _query_str) -> list:
    """
    This function retrieves page numbers that contain a keyphrase extracted from a query string 
    retrieved using a BM25 retriever. The page numbers are sorted in ascending order.

    Parameters:
    _vector_docstore (Docstore): The document store containing the vectorized nodes.
    _similarity_top_k (int): The number of top similar nodes to retrieve.
    _query_str (str): The query string from which to extract the keyphrase.

    Returns:
    list: A list of page numbers containing the keyphrase, sorted in ascending order.

    The function works by first initializing a BM25Retriever object with the provided document store 
    and similarity_top_k. 
    It then extracts a keyphrase from the query string using a keyBURT model.
    Finally, it uses the BM25Retriever object and the extracted keyphrase to retrieve page numbers 
    containing the keyphrase, which are then returned as a list.
    """

    _bm25_retriever = BM25Retriever.from_defaults(
        similarity_top_k=_similarity_top_k,
        docstore=_vector_docstore,
        )

    # Get keyphrase from the query string using the keyBURT model
    _query_keyphrase = get_keyphrase_from_query_str(_query_str)
    print(f"\n\nThe keyphrase is: {_query_keyphrase}\n")

    # Get page numbers containing the keyphrase using bm25 model, sorted in ascending order
    _page_numbers = get_page_numbers_using_bm25_and_keyphrase(_bm25_retriever,
                                                            _query_keyphrase
                                                            )
    return _page_numbers



def get_fusion_accumulate_sort_detail_engine(
                                    _vector_index,
                                    _vector_docstore,
                                    _similarity_top_k,
                                    _num_queries,
                                    _fusion_top_n,
                                    ):
    
    _fusion_retriever = get_fusion_retriever(
                                        _vector_index,
                                        _vector_docstore,
                                        _similarity_top_k,
                                        _num_queries,
                                        _fusion_top_n,
                                        )

    # Create an accumulate, fusion, and sort engine
    _accumulate_fusion_sort_engine = RetrieverQueryEngine.from_args(
                                            retriever=_fusion_retriever, 
                                            node_postprocessors=[PageSortNodePostprocessor()],
                                            response_mode="accumulate",
                                            )

    _fusion_accumulate_sort_detail_engine = change_accumulate_engine_prompt_to_in_detail(
                                                            _accumulate_fusion_sort_engine
                                                            )
    
    return _fusion_accumulate_sort_detail_engine


def get_fusion_accumulate_sort_detail_tool(
                                    _vector_index,
                                    _vector_docstore,
                                    _similarity_top_k,
                                    _num_queries,
                                    _fusion_top_n,
                                    ):
    
    _fusion_retriever = get_fusion_retriever(
                                        _vector_index,
                                        _vector_docstore,
                                        _similarity_top_k,
                                        _num_queries,
                                        _fusion_top_n,
                                        )

    # Create an accumulate, fusion, and sort engine
    _accumulate_fusion_sort_engine = RetrieverQueryEngine.from_args(
                                            retriever=_fusion_retriever, 
                                            node_postprocessors=[PageSortNodePostprocessor()],
                                            response_mode="accumulate",
                                            )

    _fusion_accumulate_sort_detail_engine = change_accumulate_engine_prompt_to_in_detail(
                                                            _accumulate_fusion_sort_engine
                                                            )

    _fusion_accumulate_sort_detail_tool = QueryEngineTool.from_defaults(
        name="fusion_tool",
        query_engine=_fusion_accumulate_sort_detail_engine,
        description=(
            "Useful for retrieving specific context from the document."
        ),
    )

    return _fusion_accumulate_sort_detail_tool


def get_summary_tree_detail_tool(
                            _summary_description: str,
                            _storage_context_summary: StorageContext,
                            ):
    
    # Create suammry engine
    _summary_tree_detail_engine = get_summary_tree_detail_engine(
                                                            _storage_context_summary
                                                            )

    # Summary tool
    _summary_tree_detail_tool = QueryEngineTool.from_defaults(
        name="summary_tool",
        query_engine=_summary_tree_detail_engine,
        description=_summary_description,
    )

    return _summary_tree_detail_tool


def get_fusion_accumulate_keyphrase_sort_detail_tool(
                                            _vector_index: VectorStoreIndex,
                                            _similarity_top_k: int,
                                            _page_numbers: list,
                                            _fusion_top_n: int,
                                            _query_str: str,
                                            _num_queries: int,
                                            _rerank: SentenceTransformerRerank
                                            ) -> QueryEngineTool:
    """
    This function creates a QueryEngineTool that uses a fusion accumulate filter sort detail 
    engine. The engine is built using a vector retriever and a BM25 filter retriever. The BM25 
    filter retriever is built using the vector filter retriever and a query string. The fusion 
    accumulate filter sort detail engine is then created using the vector retriever, the BM25 
    filter retriever, the fusion top n, the number of queries, and a rerank object.

    Args:
        _vector_index (VectorStoreIndex): The vector store index.
        _similarity_top_k (int): The number of similar nodes to retrieve.
        _page_numbers (list): The page numbers to filter the nodes by.
        _fusion_top_n (int): The number of nodes to return from the fusion engine.
        _query_str (str): The query string to use for the BM25 filter retriever.
        _num_queries (int): The number of queries to use for the fusion engine.
        _rerank (SentenceTransformerRerank): The rerank object to use for the fusion engine.

    Returns:
        QueryEngineTool: A QueryEngineTool that uses the fusion accumulate filter sort detail 
        engine.
    """

    # Create vector retreiver (against a query) with metadata filter using page numbers
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

    # Create vector retreiver (against a query) with metadata filter using page numbers
    _vector_retriever = _vector_index.as_retriever(
                                    similarity_top_k=_similarity_top_k,
                                )

    # Get bm25 filter retriever to build a fusion engine with metadata filter 
    # (query_str is for getting the nodes first)
    _bm25_filter_retriever = get_bm25_filter_retriever(
                                                    _vector_filter_retriever, 
                                                    _query_str, 
                                                    _similarity_top_k
                                                    )

    # Get fusion accumulate filter sort detail engine using _bm25_filter_retriever 
    # and _vector_retriever (Note: _vector_filter_retriever is not used in this engine)
    _fusion_accumulate_filter_sort_detail_engine = get_fusion_accumulate_filter_sort_detail_engine(
                                                                        # _vector_filter_retriever,
                                                                        _vector_retriever,
                                                                        _bm25_filter_retriever,
                                                                        _fusion_top_n,
                                                                        _num_queries,
                                                                        _rerank
                                                                        )

    _fusion_accumulate_keyphrase_sort_detail_tool = QueryEngineTool.from_defaults(
        name="fusion_keyphrase_tool",
        query_engine=_fusion_accumulate_filter_sort_detail_engine,
        description=(
            "Useful for retrieving specific context from the document."
        ),
    )

    return _fusion_accumulate_keyphrase_sort_detail_tool


def get_fusion_tree_keyphrase_sort_detail_tool(
                                        _vector_index: VectorStoreIndex,
                                        _vector_docstore: MongoDocumentStore,
                                        _similarity_top_k: int,
                                        _page_numbers: list,
                                        _fusion_top_n: int,
                                        _query_str: str,
                                        _num_queries: int,
                                        _rerank: SentenceTransformerRerank,
                                        _specific_tool_description: str,
                                        ) -> QueryEngineTool:
    """
    This function creates a QueryEngineTool that uses a fusion tree filter sort detail engine.
    The engine is built using a vector retriever and a BM25 filter retriever. The BM25 
    filter retriever is built using the vector filter retriever and a query string. The fusion 
    tree filter sort detail engine is then created using the vector retriever, the BM25 
    filter retriever, the fusion top n, the number of queries, and a rerank object.

    Args:
        _vector_index (VectorStoreIndex): The vector store index.
        _similarity_top_k (int): The number of similar nodes to retrieve.
        _page_numbers (list): The page numbers to filter the nodes by.
        _fusion_top_n (int): The number of nodes to return from the fusion engine.
        _query_str (str): The query string to use for the BM25 filter retriever.
        _num_queries (int): The number of queries to use for the fusion engine.
        _rerank (SentenceTransformerRerank): The rerank object to use for the fusion engine.

    Returns:
        QueryEngineTool: A QueryEngineTool that uses the fusion tree filter sort detail 
        engine.
    """

    # Create vector retreiver (against a query) with metadata filter using page numbers
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

    # Create vector retreiver (against a query) with metadata filter using page numbers
    _vector_retriever = _vector_index.as_retriever(
                                    similarity_top_k=_similarity_top_k,
                                )

    # Get bm25 filter retriever to build a fusion engine with metadata filter 
    # (query_str is for getting the nodes first)
    _bm25_filter_retriever = get_bm25_filter_retriever(
                                                    _vector_filter_retriever, 
                                                    _query_str, 
                                                    _similarity_top_k
                                                    )

    # Get fusion tree filter sort detail engine using _bm25_filter_retriever 
    # and _vector_retriever (Note: _vector_filter_retriever is not used in this engine)
    _fusion_tree_filter_sort_detail_engine = get_fusion_tree_filter_sort_detail_engine(
                                                                        # _vector_filter_retriever,
                                                                        _vector_retriever,
                                                                        _vector_docstore,
                                                                        _bm25_filter_retriever,
                                                                        _fusion_top_n,
                                                                        _num_queries,
                                                                        _rerank
                                                                        )   

    _fusion_tree_keyphrase_sort_detail_tool = QueryEngineTool.from_defaults(
        name="fusion_keyphrase_tool",
        query_engine=_fusion_tree_filter_sort_detail_engine,
        description=_specific_tool_description,
    )

    return _fusion_tree_keyphrase_sort_detail_tool


def get_fusion_accumulate_page_filter_sort_detail_engine(
                                            _vector_filter_retriever,
                                            _similarity_top_k_filter: int,
                                            _fusion_top_n_filter: int,
                                            _query_str: str,
                                            _num_queries_filter: int,
                                            ):

    # Get bm25 filter retriever to build a fusion engine with metadata filter (query_str is for getting the nodes first)
    _bm25_filter_retriever = get_bm25_filter_retriever(
                                                _vector_filter_retriever, 
                                                _query_str, 
                                                _similarity_top_k_filter
                                                )

    # Get fusion accumulate filter sort detail engine
    _fusion_accumulate_filter_sort_detail_engine = get_fusion_accumulate_filter_sort_detail_engine(
                                                                        _vector_filter_retriever,
                                                                        _bm25_filter_retriever,
                                                                        _fusion_top_n_filter,
                                                                        _num_queries_filter
                                                                        )

    return _fusion_accumulate_filter_sort_detail_engine


def get_fusion_tree_page_filter_sort_detail_engine(
                                        _vector_filter_retriever,
                                        _similarity_top_k_filter: int,
                                        _fusion_top_n_filter: int,
                                        _query_str: str,
                                        _num_queries_filter: int,
                                        ):

    # Get bm25 filter retriever to build a fusion engine with metadata filter (query_str is for getting the nodes first)
    _bm25_filter_retriever = get_bm25_filter_retriever(
                                                _vector_filter_retriever, 
                                                _query_str, 
                                                _similarity_top_k_filter
                                                )

    # Get fusion tree filter sort detail engine
    _fusion_tree_filter_sort_detail_engine = get_fusion_tree_filter_sort_detail_engine(
                                                                        _vector_filter_retriever,
                                                                        _bm25_filter_retriever,
                                                                        _fusion_top_n_filter,
                                                                        _num_queries_filter
                                                                        )

    return _fusion_tree_filter_sort_detail_engine


def change_default_engine_prompt_to_in_detail(engine):

    new_text_qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query. \n"
    "Try to include as many key details as possible.\n"
    "Query: {query_str}\n"
    "Answer: "
    )

    new_refine_tmpl_str = (
    "The original query is as follows:\n"
    "---------------------\n"
    "{query_str}\n"
    "---------------------\n"
    "We have provided an existing answer:\n"
    "---------------------\n"
    "{existing_answer}\n"
    "---------------------\n"
    "We have the opportunity to refine the existing answer (only if needed) \n"
    "with some more context below.\n"
    "---------------------\n"
    "{context_msg}\n"
    "---------------------\n"
    "Given the new context, refine the original answer to better answer the query. \n"
    "Try to include as many key details as possible. \n"
    "If the context isn't useful, return the original answer.\n"
    "Refined Answer: "
    )

    new_text_qa_tmpl = PromptTemplate(new_text_qa_tmpl_str)
    engine.update_prompts(
                    {"response_synthesizer:text_qa_template": 
                                            new_text_qa_tmpl}
                                            )
    
    new_refine_tmpl = PromptTemplate(new_refine_tmpl_str)
    engine.update_prompts(
                    {"response_synthesizer:refine_template": 
                                            new_refine_tmpl}
                                            )
    return engine


def change_tree_engine_prompt_to_in_detail(engine):

    new_summary_tmpl_str = (
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, "
    "answer the query below.\n"
    "Try to include as many key details as possible.\n"
    "Query: {query_str}\n"
    "Answer: "
    )

    new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    engine.update_prompts(
                    {"response_synthesizer:summary_template": 
                                            new_summary_tmpl}
                                            )
    return engine


def change_accumulate_engine_prompt_to_in_detail(engine):

    new_text_qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query. \n"
    "Try to include as many key details as possible.\n"
    "Query: {query_str}\n"
    "Answer: "
    )

    new_text_qa_tmpl = PromptTemplate(new_text_qa_tmpl_str)
    engine.update_prompts(
                    {"response_synthesizer:text_qa_template": 
                                            new_text_qa_tmpl}
                                            )
    return engine


def change_summary_engine_prompt_to_in_detail(engine):

    new_summary_tmpl_str = (
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, "
    "answer the query below.\n"
    "Try to provide a detailed summary that captures the major story line, "
    "main arguments, supporting evidence, and key conclusions in at least 500 words.\n"
    "Query: {query_str}\n"
    "Answer: "
    )

    new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    engine.update_prompts(
                    {"response_synthesizer:summary_template": 
                                            new_summary_tmpl}
                                            )
    return engine


def display_prompt_dict(type, _prompts_dict):
    print(f"\n{type}\n")
    for k, p in _prompts_dict.items():
        print(f"\nPrompt Key: {k} \nText:\n")
        print(p.get_template() + "\n")


def print_retreived_nodes(method, _retrieved_nodes):

    print(f"\n\nMETHOD: {method.upper()}")
    # Loop through each NodeWithScore in the retreived nodes
    for (i, node_with_score) in enumerate(_retrieved_nodes):
        _node = node_with_score.node  # The TextNode object
        score = node_with_score.score  # The similarity score
        chunk_id = _node.id_  # The chunk ID

        # Extract the relevant metadata from the node
        file_name = _node.metadata.get("file_name", "Unknown")
        file_path = _node.metadata.get("file_path", "Unknown")

        # Extract the text content from the node
        text_content = _node.text if _node.text else "No content available"

        # Print the results in a user-friendly format
        print(f"\n\n{method.upper()}:")
        print(f"Item number: {i+1}")
        print(f"Score: {score}")
        # print(f"File Name: {file_name}")
        # print(f"File Path: {file_path}")
        print(f"Id: {chunk_id}")
        print("\nExtracted Content:\n")
        print(text_content)
        # print("\n" + "=" * 40 + " End of Result " + "=" * 40 + "\n")
        # print("\n")