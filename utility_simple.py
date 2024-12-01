from typing import List, Optional, Union

from keybert import KeyBERT
from llama_index.core import (
                        PromptTemplate,
                        QueryBundle,
                        StorageContext,
                        SummaryIndex,
                        VectorStoreIndex
                        )
from llama_index.core.indices.postprocessor import (
                        PrevNextNodePostprocessor,
                        )
from llama_index.core.retrievers import (
                        QueryFusionRetriever, 
                        BaseRetriever,
                        )
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
                        NodeWithScore, 
                        TextNode,
                        )
from llama_index.core.tools import QueryEngineTool
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.vector_stores.milvus import MilvusVectorStore


class PageSortNodePostprocessor(BaseNodePostprocessor):
    """
    A custom node postprocessor that sorts nodes based on page number and the order they appear in a document.

    Attributes:
        None

    Methods:
        _postprocess_nodes: Sorts nodes based on the page they appear on and the order they appear in that page.
    """
    def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle]
            ) -> List[NodeWithScore]:
        """
        Sorts nodes based on the page they appear on and the order they appear in that page.
        (IMPORTANT: ALTHOUGH QUERY_BUNDLE IS NOT USED IN THIS METHOD, IT IS REQUIRED FOR THE INTERFACE.
        IF REMOVED WILL CAUSE AN ERROR OF TOO MANY POSITIONAL ARGUMENTS.)
        Args:
            nodes (List[NodeWithScore]): A list of nodes to be sorted.

        Returns:
            List[NodeWithScore]: A list of nodes sorted based on the page they appear on and the order they appear 
            in that page.
        """

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


def get_article_link(article_dir, article_name):
    """
    This function takes in a directory and an article name,
    and returns the full path to the article.

    Parameters:
    article_dir (str): The directory where the article is located.
    article_name (str): The name of the article.

    Returns:
    str: The full path to the article.
    """

    return f"./data/{article_dir}/{article_name}"


def get_database_and_sentence_splitter_collection_name(
        article_directory: str,
        chunk_method: str,
        embed_model_name: str,
        chunk_size: int,
        chunk_overlap: int,
        metadata: Optional[str] = None,
        ) -> tuple:
    """
    Generate names for the database, collection, and summary collection based on the given 
    parameters.

    Parameters:
    article_directory (str): The directory where the articles are stored.
    chunk_method (str): The method used for chunking the text.
    embed_model_name (str): The name of the embedding model.
    chunk_size (int): The size of each chunk.
    chunk_overlap (int): The overlap between chunks.
    metadata (Optional[str], optional): Any additional metadata to be included in the 
    collection name. Defaults to None.

    Returns:
    tuple: A tuple containing the names of the database, collection, and summary collection.
    """

    # Generate the database name
    database_name = f"{article_directory}_{chunk_method}"

    # Generate the base collection name
    base_collection_name = \
        f"{embed_model_name}_chunk_size_{chunk_size}_chunk_overlap_{chunk_overlap}"

    # Generate the collection name with optional metadata
    collection_name = f"{base_collection_name}_metadata_{metadata}" \
        if metadata else base_collection_name

    # Generate the summary collection name with optional metadata
    collection_name_summary = f"{base_collection_name}_summary_metadata_{metadata}" \
        if metadata else f"{base_collection_name}_summary"

    return database_name, collection_name, collection_name_summary


def get_vector_store_docstore_and_storage_context(
        uri_milvus: str, 
        uri_mongo: str, 
        database_name: str, 
        collection_name_vector: str, 
        embed_model_dim: int
        ):
    """
    This function initializes a vector store, a MongoDB vector document store,
    and a vector storage context using the provided URIs, database name,
    collection name, and embedding model dimension.

    Parameters:
    uri_milvus (str): The URI for the Milvus server.
    uri_mongo (str): The URI for the MongoDB server.
    database_name (str): The name of the database.
    collection_name_vector (str): The name of the collection for vector data.
    embed_model_dim (int): The dimension of the embedding model.

    Returns:
    tuple: A tuple containing the initialized vector store,
           vector document store, and vector storage context.
    """

    # Initialize vector store (a new empty collection will be created in Milvus server)
    vector_store = MilvusVectorStore(
        uri=uri_milvus,
        db_name=database_name,
        collection_name=collection_name_vector,
        dim=embed_model_dim,  # dim of HuggingFace "BAAI/bge-small-en-v1.5" embedding model
        enable_dynamic_field=False,
    )

    # Initialize MongoDB vector document store (Not yet saved to MongoDB server)
    vector_docstore = MongoDocumentStore.from_uri(
        uri=uri_mongo,
        db_name=database_name,
        namespace=collection_name_vector,
    )

    # Initialize vector storage context: use Milvus as vector store and Mongo as document store
    storage_context_vector = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=vector_docstore,
    )

    return vector_store, vector_docstore, storage_context_vector


def get_summary_storage_context(
        uri_mongo: str, 
        database_name: str, 
        collection_name_summary: str
        ) -> Union[StorageContext, None]:
    """
    This function creates a storage context for summaries using MongoDB as the document store.

    Parameters:
    uri_mongo (str): The URI for the MongoDB server.
    database_name (str): The name of the MongoDB database to be used.
    collection_name_summary (str): The name of the MongoDB collection to store the summaries.

    Returns:
    Union[StorageContext, None]: A storage context object that uses MongoDB as the document 
    store, or None if there is an error initializing the document store.
    """

    # Initiate MongoDB summary docstore (Not yet saved to MongoDB server)
    try:
        summary_docstore = MongoDocumentStore.from_uri(
            uri=uri_mongo,
            db_name=database_name,
            namespace=collection_name_summary
        )
    except Exception as e:
        print(f"Error initializing MongoDB document store: {e}")
        return None

    # Initiate summary storage context: use Milvus as vector store and Mongo as docstore
    storage_context_summary = StorageContext.from_defaults(
        docstore=summary_docstore
    )

    return storage_context_summary


def get_summary_tree_detail_engine(storage_context_summary):
    """
    This function creates a summary tree detail engine.

    Parameters:
    storage_context_summary (object): A storage context summary object.

    Returns:
    tree_summary_detail_engine (object): A retriever query engine that provides detailed summary trees.
    """
    # Extract nodes from the storage context summary
    extracted_nodes = list(storage_context_summary.docstore.docs.values())

    # Create a summary index using the extracted nodes
    summary_index = SummaryIndex(nodes=extracted_nodes)

    # Create a retriever from the summary index
    summary_retriever = summary_index.as_retriever()

    # Create a retriever query engine that uses the summary retriever and tree summarize response mode
    tree_summary_engine = RetrieverQueryEngine.from_args(
        retriever=summary_retriever,
        response_mode="tree_summarize",
        # node_postprocessors=[PageSortNodePostprocessor()],
        use_async=True,
    )

    # Modify the tree summary engine prompt to provide more detailed summaries
    tree_summary_detail_engine = change_summary_engine_prompt_to_in_detail(tree_summary_engine)

    return tree_summary_detail_engine


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


def get_keyphrase_from_query_str(query_str: str) -> str:
    """
    Extract keyphrases from a given query string using the KeyBERT model.

    Parameters:
    query_str (str): The input query string.

    Returns:
    str: A string containing the extracted keyphrases.
    """

    # Initialize KeyBERT model
    kw_model = KeyBERT()

    # Extract keywords (keyphrases) from the query string
    keywords = kw_model.extract_keywords(
        docs=query_str,
        keyphrase_ngram_range=(3, 3),
        top_n=5,
    )

    # Create a set to store unique keywords
    keywords_set = set()
    for k in keywords:
        keywords_set.update(k[0].split())

    # Concatenate the keywords into a single string
    final_keywords = " ".join(keywords_set)

    # Return the final keyphrase string
    return final_keywords


# def get_keyphrase_from_query_str(query_str: str) -> str:

#     # Initialize KeyBERT model
#     kw_model = KeyBERT()

#     # Extract keywords (keyphrases) from the query string
#     keywords = kw_model.extract_keywords(
#                                 docs=query_str, 
#                                 keyphrase_ngram_range=(3, 3), 
#                                 # use_maxsum=True,
#                                 # nr_candidates=5,
#                                 top_n=3,
#                                 )
#     keywords_set = set()
#     for k in keywords:
#         keywords_set.update(k[0].split())

#     final_keywords = str()
#     for i in keywords_set:
#         final_keywords = final_keywords + " " + i

#     # Remove leading whitespace and return the final keyphrase
#     return final_keywords.lstrip()  


# def get_text_nodes_using_bm25_and_keyphrase(bm25_retriever, keywords) -> list:
#     """
#     This function retrieves a list of page numbers of the nodes that contain a given keyphrase in a 
#     document using BM25 retriever. The page numbers are sorted in ascending order.

#     Parameters:
#     bm25_retriever (BM25Retriever): An instance of the BM25Retriever class.
#     keywords (str): The keyword to search for in the documents.

#     Returns:
#     list: A list of page numbers (as strings) that contain the keyphrase, sorted in ascending order.
#     """

#     bm25_score_nodes = bm25_retriever.retrieve(keywords)
#     bm25_text_nodes = [node.node for node in bm25_score_nodes ]  # get TextNode from ScoredNode

#     return bm25_text_nodes


# def get_page_numbers_using_bm25_and_keyphrase(bm25_retriever, keywords) -> list:
#     """
#     This function retrieves a list of page numbers of the nodes that contain a given keyphrase in a 
#     document using BM25 retriever. The page numbers are sorted in ascending order.

#     Parameters:
#     bm25_retriever (BM25Retriever): An instance of the BM25Retriever class.
#     keywords (str): The keyword to search for in the documents.

#     Returns:
#     list: A list of page numbers (as strings) that contain the keyphrase, sorted in ascending order.
#     """

#     bm25_score_nodes = bm25_retriever.retrieve(keywords)
#     bm25_text_nodes = [node.node for node in bm25_score_nodes ]  # get TextNode from ScoredNode

#     # page_numbers = sorted([int(n.metadata["source"]) for n in bm25_text_nodes])  # get and sort the page numbers
#     page_numbers = [int(n.metadata["source"]) for n in bm25_text_nodes]  # get and sort the page numbers
#     page_numbers = [str(p) for p in page_numbers]  # convert back to str

#     return page_numbers



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


def get_bm25_filter_retriever(
        vector_filter_retriever: BaseRetriever, 
        query: str, 
        # vector_docstore: Optional[MongoDocumentStore] = None,
        ) -> BM25Retriever:
    """
    This function creates a BM25 filter retriever using nodes retrieved from a vector retriever
    and a query string.

    Parameters:
    vector_filter_retriever (BaseRetriever): An instance of a vector retriever class that 
                                        supports retrieving nodes based on a query string.
    query (str): The query string used to retrieve nodes from the vector retriever.
    num_top_results (int): The number of top similar nodes to consider for the BM25 filter retriever.

    Returns:
    bm25_filter_retriever (BM25Retriever): An instance of BM25Retriever class that uses the
                                           retrieved nodes from the vector retriever on a query string.
    """
    try:
        # Retrieve nodes for BM25 filter retriever using the vector filter retriever and the query
        scored_nodes = vector_filter_retriever.retrieve(query)

        # Extract TextNodes from NodeWithScore objects
        text_nodes = [scored_node.node for scored_node in scored_nodes]

        print(f"Text nodes in page Bm25 length is: {len(text_nodes)}")
        for i, n in enumerate(text_nodes):
            print(f"Item {i+1} of the text nodes in page Bm25 is page: {n.metadata['source']}")
    except Exception as e:
        print(f"Error in retrieving nodes for BM25 page filter retriever: {e}")
        return None

    # # Create simple BM25 filter 
    # bm25_retriever= BM25Retriever.from_defaults(
    #     similarity_top_k=36,
    #     docstore=vector_docstore
    # )

    # # Retrieve nodes for BM25 retriever
    # scored_nodes_raw = bm25_retriever.retrieve(query)

    # # Extract TextNodes from NodeWithScore objects
    # text_nodes_raw = [scored_node.node for scored_node in scored_nodes_raw]

    # print(f"keyphrase RAW Bm25 retrieved nodes length is: {len(text_nodes_raw)}")
    # for i, n in enumerate(text_nodes_raw):
    #     print(f"Item {i+1} of the RAW keyphrase Bm25 retrieved nodes is page: {n.metadata['source']}")


    # Create BM25 filter retriever using the extracted TextNodes
    bm25_filter_retriever= BM25Retriever.from_defaults(
        similarity_top_k=len(text_nodes),
        nodes=text_nodes,
    )

    # Retrieve nodes for BM25 filter retriever using the vector filter retriever and the query
    scored_nodes = bm25_filter_retriever.retrieve(query)

    # Extract TextNodes from NodeWithScore objects
    text_nodes = [scored_node.node for scored_node in scored_nodes]

    print(f"\nkeyphrase Bm25 retrieved nodes length is: {len(text_nodes)}")
    for i, n in enumerate(text_nodes):
        print(f"Item {i+1} of the keyphrase Bm25 retrieved nodes is page: {n.metadata['source']}")

    return bm25_filter_retriever


# def get_bm25_filter_retriever(_vector_filter_retriever, 
#                               _query_str: str, 
#                               _similarity_top_k: int)-> BM25Retriever:
#     """
#     This function creates a BM25 filter retriever using the nodes retrieved from a vector 
#     filter retriever and a query string.

#     Parameters:
#     _vector_filter_retriever (BaseRetriever): An instance of a vector retriever class that 
#                                         supports retrieving nodes based on a query string.
#     _query_str (str): The query string used to retrieve nodes from the vector filter retriever.
#     _similarity_top_k (int): The number of top similar nodes to consider for the BM25 filter 
#                             retriever.

#     Returns:
#     _bm25_filter_retriever (BM25Retriever): An instance of a BM25Retriever class that uses the retrieved filtered nodes 
#     from the vector filter retriever on a query string.
#     """

#     # Get nodes for bm25 filter retriever
#     score_nodes = _vector_filter_retriever.retrieve(_query_str)
#     text_nodes = [node.node for node in score_nodes ]  # get TextNode from ScoredNode

#     # Create bm25 filter engine using filtered nodes
#     _bm25_filter_retriever = BM25Retriever.from_defaults(
#                                 similarity_top_k=_similarity_top_k,
#                                 nodes=text_nodes,
#                                 )

#     return _bm25_filter_retriever


# def get_fusion_accumulate_filter_sort_detail_engine(_vector_retriever,
#                                                     _bm25_filter_retriever: BM25Retriever,
#                                                     _fusion_top_n: int,
#                                                     _num_queries: int,
#                                                     _rerank: SentenceTransformerRerank,
#                                                     ):
#     """
#     This function creates a fusion filter retriever and engine that combines the results of two 
#     retrievers: a vector filter retriever and a BM25 filter retriever. The results are then sorted 
#     and reranked using a SentenceTransformerRerank object. The engine is configured to accumulate 
#     results and provide a detailed response.

#     Args:
#     _vector_filter_retriever: The first retriever to be used in the fusion.
#     _bm25_filter_retriever (BM25Retriever): The second retriever to be used in the fusion.
#     _fusion_top_n (int): The number of top results to consider from each retriever.
#     _num_queries (int): The number of queries to generate. Set this to 1 to disable query generation.
#     _rerank (SentenceTransformerRerank): The reranking model to be used.

#     Returns:
#     RetrieverQueryEngine: A query engine that uses the fusion filter retriever and the specified reranking model.
#     The engine is configured to accumulate results and provide a detailed response.
#     """    
#     # Create fusion filter retreiver and engine
#     _fusion_filter_retriever = QueryFusionRetriever(
#                                 retrievers=[
#                                         _vector_retriever, 
#                                         _bm25_filter_retriever
#                                         ],
#                                 similarity_top_k=_fusion_top_n,
#                                 num_queries=_num_queries,  # set this to 1 to disable query generation
#                                 mode="reciprocal_rerank",
#                                 retriever_weights=[0.5, 0.5],
#                                 use_async=True,
#                                 verbose=True,
#                                 # query_gen_prompt="...",  # for overriding the query generation prompt
#                                 )

#     _fusion_accumulate_filter_sort_engine = RetrieverQueryEngine.from_args(
#                                             retriever=_fusion_filter_retriever, 
#                                             node_postprocessors=[
#                                                 _rerank,
#                                                 PageSortNodePostprocessor(),
#                                                 ],
#                                             response_mode="accumulate",
#                                             )

#     _fusion_accumulate_filter_sort_detail_engine = change_accumulate_engine_prompt_to_in_detail(
#                                                             _fusion_accumulate_filter_sort_engine
#                                                             )
    
#     return _fusion_accumulate_filter_sort_detail_engine


def get_fusion_tree_filter_sort_detail_engine(
    vector_filter_retriever,
    bm25_filter_retriever: BM25Retriever,
    fusion_top_n: int,
    num_queries: int,
    rerank: BaseNodePostprocessor,
    vector_docstore: MongoDocumentStore,
    page_numbers: Optional[List[str]]=None,
):
    """
    Build a fusion tree filter sort detail engine using vector filter retriever and bm25 filter retriever.

    Args:
        vector_filter_retriever: Vector filter retriever object.
        bm25_filter_retriever (BM25Retriever): BM25 filter retriever object.
        fusion_top_n (int): The number of top documents to consider for fusion.
        num_queries (int): The number of queries to generate for fusion.

    Returns:
        A fusion tree filter sort detail engine object.
    """

    # Create fusion filter retriever
    fusion_filter_retriever = QueryFusionRetriever(
        retrievers=[
            vector_filter_retriever, 
            bm25_filter_retriever
            ],
        similarity_top_k=fusion_top_n,
        num_queries=num_queries,
        mode="reciprocal_rerank",
        retriever_weights=[0.5, 0.5],
        use_async=True,
        verbose=True,
    )

    PrevNext = PrevNextNodePostprocessor(
                                docstore=vector_docstore,
                                num_nodes=2,  # each page now has two nodes, one with next, the other previous
                                mode="both",
                                verbose=True,
                                )  # can only retrieve the two nodes on one page
    
    if page_numbers is not None:  # remove rerank since some pages may be removed
        node_postprocessors = [
                PrevNext,
                PageSortNodePostprocessor(),
                ]
    else:
        node_postprocessors = [
                            rerank,     
                            PrevNext,
                            PageSortNodePostprocessor(),
                            ]

    # Create fusion tree filter sort engine
    fusion_tree_filter_sort_engine = RetrieverQueryEngine.from_args(
        retriever=fusion_filter_retriever,
        node_postprocessors=node_postprocessors,
        response_mode="tree_summarize",
    )

    # Modify the prompt to provide detailed results
    # if page_numbers is not None:
    fusion_tree_filter_sort_detail_engine = change_tree_engine_prompt_to_in_detail(
                                                        fusion_tree_filter_sort_engine,
                                                        page_numbers,
                                                        )
    # else:
    #     fusion_tree_filter_sort_detail_engine = change_tree_engine_prompt_to_in_detail(
    #                                                         fusion_tree_filter_sort_engine
    #                                                         )

    return fusion_tree_filter_sort_detail_engine


# def get_fusion_tree_filter_sort_detail_engine(
#                                         _vector_filter_retriever,
#                                         _bm25_filter_retriever: BM25Retriever,
#                                         _fusion_top_n: int,
#                                         _num_queries: int,
#                                         ):

#     # Create fusion filter retreiver and engine
#     _fusion_filter_retriever = QueryFusionRetriever(
#                                 retrievers=[
#                                         _vector_filter_retriever, 
#                                         _bm25_filter_retriever
#                                         ],
#                                 similarity_top_k=_fusion_top_n,
#                                 # similarity_top_k=2,
#                                 num_queries=_num_queries,  # set this to 1 to disable query generation
#                                 mode="reciprocal_rerank",
#                                 retriever_weights=[0.5, 0.5],
#                                 use_async=True,
#                                 verbose=True,
#                                 # query_gen_prompt="...",  # for overriding the query generation prompt
#                                 )

#     node_postprocessors = [
#                         PageSortNodePostprocessor(),
#                         ]


#     _fusion_tree_filter_sort_engine = RetrieverQueryEngine.from_args(
#                                             retriever=_fusion_filter_retriever, 
#                                             node_postprocessors=node_postprocessors,
#                                             response_mode="tree_summarize",
#                                             )

#     _fusion_tree_filter_sort_detail_engine = change_tree_engine_prompt_to_in_detail(
#                                                             _fusion_tree_filter_sort_engine
#                                                             )
    
#     return _fusion_tree_filter_sort_detail_engine


def get_fusion_tree_keyphrase_filter_sort_detail_engine(
                                                vector_retriever,
                                                vector_docstore: MongoDocumentStore,
                                                bm25_retriever: BM25Retriever,
                                                fusion_top_n: int,
                                                num_queries: int,
                                                rerank: ColbertRerank = None,
                                                ):
    """
    This function creates a fusion filter retriever and engine that combines the results of two 
    retrievers: a vector filter retriever and a BM25 filter retriever. The results are then sorted 
    and reranked using a SentenceTransformerRerank object. The engine is configured to provide 
    a detailed response.

    Args:
    vector_filter_retriever: The first retriever to be used in the fusion.
    bm25_filter_retriever (BM25Retriever): The second retriever to be used in the fusion.
    fusion_top_n (int): The number of top results to consider from each retriever.
    num_queries (int): The number of queries to generate. Set this to 1 to disable query generation.
    rerank (SentenceTransformerRerank): The reranking model to be used.

    Returns:
    RetrieverQueryEngine: A query engine that uses the fusion filter retriever and the specified reranking model.
    The engine is configured to tree results and provide a detailed response.
    """    

    # Create fusion filter retreiver and engine
    fusion_filter_retriever = QueryFusionRetriever(
                                retrievers=[
                                        vector_retriever, 
                                        bm25_retriever
                                        ],
                                similarity_top_k=fusion_top_n,
                                # similarity_top_k=2,
                                num_queries=num_queries,  # set this to 1 to disable query generation
                                # mode="reciprocal_rerank",
                                mode="relative_score",
                                retriever_weights=[0.5, 0.5],
                                use_async=True,
                                verbose=True,
                                # query_gen_prompt="...",  # for overriding the query generation prompt
                                )

    # AutoPrevNext = AutoPrevNextNodePostprocessor(
    #                                     docstore=vector_docstore,
    #                                     num_nodes=5,
    #                                     verbose=True,
    #                                     )

    PrevNext = PrevNextNodePostprocessor(
                                    docstore=vector_docstore,
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
    
    if rerank is not None:
        node_postprocessors.insert(0, rerank)

    fusion_tree_filter_sort_engine = RetrieverQueryEngine.from_args(
                                            retriever=fusion_filter_retriever, 
                                            node_postprocessors=node_postprocessors,
                                            response_mode="tree_summarize",
                                            )

    fusion_tree_filter_sort_detail_engine = change_tree_engine_prompt_to_in_detail(
                                                            fusion_tree_filter_sort_engine
                                                            )
    
    return fusion_tree_filter_sort_detail_engine


def get_text_nodes_from_query_keyphrase(
        vector_docstore,
        similarity_top_n: int,
        query_str: str
        ) -> List[int]:
    """
    This function takes a vectorized document store, a similarity top-k value, and a query string.
    It creates a BM25 retriever, extracts a keyphrase from the query string, and uses the BM25 retriever
    and the keyphrase to get a list of page numbers containing the keyphrase, sorted in ascending order.

    :param vector_docstore: A vectorized document store.
    :param similarity_top_k: An integer value representing the number of results to return.
    :param query_str: A string value representing the query.
    :return: A list of page numbers containing the keyphrase.
    """

    bm25_retriever = BM25Retriever.from_defaults(
        similarity_top_k=similarity_top_n,
        docstore=vector_docstore,
    )

    # Get keyphrase from the query string using the keyBURT model
    query_keyphrase = get_keyphrase_from_query_str(query_str)

    print(f"\nThe query in keyphrase tool is: {query_str}.")
    print(f"\nThe keyphrase is: {query_keyphrase}.")

    # Get page numbers containing the keyphrase using bm25 model, sorted in ascending order
    bm25_score_nodes = bm25_retriever.retrieve(query_keyphrase)
    bm25_text_nodes = [node.node for node in bm25_score_nodes ]  # get TextNode from ScoredNode

    return bm25_text_nodes


# def get_page_numbers_from_query_keyphrase(
#         vector_docstore,
#         similarity_top_n: int,
#         query_str: str
#         ) -> List[int]:
#     """
#     This function takes a vectorized document store, a similarity top-k value, and a query string.
#     It creates a BM25 retriever, extracts a keyphrase from the query string, and uses the BM25 retriever
#     and the keyphrase to get a list of page numbers containing the keyphrase, sorted in ascending order.

#     :param vector_docstore: A vectorized document store.
#     :param similarity_top_k: An integer value representing the number of results to return.
#     :param query_str: A string value representing the query.
#     :return: A list of page numbers containing the keyphrase.
#     """

#     bm25_retriever = BM25Retriever.from_defaults(
#         similarity_top_k=similarity_top_n,
#         docstore=vector_docstore,
#     )

#     # Get keyphrase from the query string using the keyBURT model
#     query_keyphrase = get_keyphrase_from_query_str(query_str)

#     print(f"\nThe query in keyphrase tool is: {query_str}.")
#     print(f"\nThe keyphrase is: {query_keyphrase}.")

#     # Get page numbers containing the keyphrase using bm25 model, sorted in ascending order
#     page_numbers = get_page_numbers_using_bm25_and_keyphrase(
#         bm25_retriever, 
#         query_keyphrase)

#     print(f"\nThe keyphrase page numbers are: {page_numbers}.")
#     return page_numbers


# def get_page_numbers_from_query_keyphrase(
#         _vector_docstore, 
#         _similarity_top_k, 
#         _query_str
#         ) -> list:

#     _bm25_retriever = BM25Retriever.from_defaults(
#         similarity_top_k=_similarity_top_k,
#         docstore=_vector_docstore,
#         )

#     # Get keyphrase from the query string using the keyBURT model
#     _query_keyphrase = get_keyphrase_from_query_str(_query_str)
#     # print(f"\n\nThe keyphrase is: {_query_keyphrase}\n")

#     # Get page numbers containing the keyphrase using bm25 model, sorted in ascending order
#     _page_numbers = get_page_numbers_using_bm25_and_keyphrase(_bm25_retriever,
#                                                             _query_keyphrase                                        )
#     return _page_numbers



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
        summary_description: str, 
        storage_context_summary: StorageContext
        ) -> QueryEngineTool:
    """
    This function creates and returns a QueryEngineTool for a given summary description and 
    storage context.

    Parameters:
    summary_description (str): A description of the summary tool.
    storage_context_summary (StorageContext): The storage context for the summary engine.

    Returns:
    QueryEngineTool: A QueryEngineTool with the name "summary_tool" and the provided 
                    description. The query_engine of this tool is a summary engine created 
                    using the provided storage context.
    """
    # Create summary engine
    summary_tree_detail_engine = get_summary_tree_detail_engine(storage_context_summary)

    # Summary tool
    summary_tree_detail_tool = QueryEngineTool.from_defaults(
        name="summary_tool",
        query_engine=summary_tree_detail_engine,
        description=summary_description,
    )

    return summary_tree_detail_tool


# def get_summary_tree_detail_tool(
#                             _summary_description: str,
#                             _storage_context_summary: StorageContext,
#                             ):
    
#     # Create suammry engine
#     _summary_tree_detail_engine = get_summary_tree_detail_engine(
#                                                             _storage_context_summary
#                                                             )

#     # Summary tool
#     _summary_tree_detail_tool = QueryEngineTool.from_defaults(
#         name="summary_tool",
#         query_engine=_summary_tree_detail_engine,
#         description=_summary_description,
#     )

#     return _summary_tree_detail_tool


# def get_fusion_accumulate_keyphrase_sort_detail_tool(
#                                             _vector_index: VectorStoreIndex,
#                                             _similarity_top_k: int,
#                                             _page_numbers: list,
#                                             _fusion_top_n: int,
#                                             _query_str: str,
#                                             _num_queries: int,
#                                             _rerank: SentenceTransformerRerank
#                                             ) -> QueryEngineTool:
#     """
#     This function creates a QueryEngineTool that uses a fusion accumulate filter sort detail 
#     engine. The engine is built using a vector retriever and a BM25 filter retriever. The BM25 
#     filter retriever is built using the vector filter retriever and a query string. The fusion 
#     accumulate filter sort detail engine is then created using the vector retriever, the BM25 
#     filter retriever, the fusion top n, the number of queries, and a rerank object.

#     Args:
#         _vector_index (VectorStoreIndex): The vector store index.
#         _similarity_top_k (int): The number of similar nodes to retrieve.
#         _page_numbers (list): The page numbers to filter the nodes by.
#         _fusion_top_n (int): The number of nodes to return from the fusion engine.
#         _query_str (str): The query string to use for the BM25 filter retriever.
#         _num_queries (int): The number of queries to use for the fusion engine.
#         _rerank (SentenceTransformerRerank): The rerank object to use for the fusion engine.

#     Returns:
#         QueryEngineTool: A QueryEngineTool that uses the fusion accumulate filter sort detail 
#         engine.
#     """

#     # Create vector retreiver (against a query) with metadata filter using page numbers
#     _vector_filter_retriever = _vector_index.as_retriever(
#                                     similarity_top_k=_similarity_top_k,
#                                     filters=MetadataFilters.from_dicts(
#                                         [{
#                                             "key": "source", 
#                                             "value": _page_numbers,
#                                             "operator": "in"
#                                         }]
#                                     )
#                                 )

#     # Create vector retreiver (against a query) with metadata filter using page numbers
#     _vector_retriever = _vector_index.as_retriever(
#                                     similarity_top_k=_similarity_top_k,
#                                 )

#     # Get bm25 filter retriever to build a fusion engine with metadata filter 
#     # (query_str is for getting the nodes first)
#     _bm25_filter_retriever = get_bm25_filter_retriever(
#                                                     _vector_filter_retriever, 
#                                                     _query_str, 
#                                                     _similarity_top_k
#                                                     )

#     # Get fusion accumulate filter sort detail engine using _bm25_filter_retriever 
#     # and _vector_retriever (Note: _vector_filter_retriever is not used in this engine)
#     _fusion_accumulate_filter_sort_detail_engine = get_fusion_accumulate_filter_sort_detail_engine(
#                                                                         # _vector_filter_retriever,
#                                                                         _vector_retriever,
#                                                                         _bm25_filter_retriever,
#                                                                         _fusion_top_n,
#                                                                         _num_queries,
#                                                                         _rerank
#                                                                         )

#     _fusion_accumulate_keyphrase_sort_detail_tool = QueryEngineTool.from_defaults(
#         name="fusion_keyphrase_tool",
#         query_engine=_fusion_accumulate_filter_sort_detail_engine,
#         description=(
#             "Useful for retrieving specific context from the document."
#         ),
#     )

#     return _fusion_accumulate_keyphrase_sort_detail_tool


# def get_fusion_tree_keyphrase_sort_detail_tool(
#                                         _vector_index: VectorStoreIndex,
#                                         _vector_docstore: MongoDocumentStore,
#                                         _similarity_top_k: int,
#                                         _page_numbers: list,
#                                         _fusion_top_n: int,
#                                         _query_str: str,
#                                         _num_queries: int,
#                                         _rerank: BaseNodePostprocessor,
#                                         _specific_tool_description: str,
#                                         ) -> QueryEngineTool:
#     """
#     This function creates a QueryEngineTool that uses a fusion tree filter sort detail engine.
#     The engine is built using a vector retriever and a BM25 filter retriever. The BM25 
#     filter retriever is built using the vector filter retriever and a query string. The fusion 
#     tree filter sort detail engine is then created using the vector retriever, the BM25 
#     filter retriever, the fusion top n, the number of queries, and a rerank object.

#     Args:
#         _vector_index (VectorStoreIndex): The vector store index.
#         _similarity_top_k (int): The number of similar nodes to retrieve.
#         _page_numbers (list): The page numbers to filter the nodes by.
#         _fusion_top_n (int): The number of nodes to return from the fusion engine.
#         _query_str (str): The query string to use for the BM25 filter retriever.
#         _num_queries (int): The number of queries to use for the fusion engine.
#         _rerank (SentenceTransformerRerank): The rerank object to use for the fusion engine.

#     Returns:
#         QueryEngineTool: A QueryEngineTool that uses the fusion tree filter sort detail 
#         engine.
#     """

#     # Create vector retreiver (against a query) with metadata filter using page numbers
#     _vector_filter_retriever = _vector_index.as_retriever(
#                                     similarity_top_k=_similarity_top_k,
#                                     filters=MetadataFilters.from_dicts(
#                                         [{
#                                             "key": "source", 
#                                             "value": _page_numbers,
#                                             "operator": "in"
#                                         }]
#                                     )
#                                 )

#     # Create vector retreiver (against a query) with metadata filter using page numbers
#     _vector_retriever = _vector_index.as_retriever(
#                                     similarity_top_k=_similarity_top_k,
#                                 )

#     # Get bm25 filter retriever to build a fusion engine with metadata filter 
#     # (query_str is for getting the nodes first)
#     _bm25_filter_retriever = get_bm25_filter_retriever(
#                                                     _vector_filter_retriever, 
#                                                     _query_str, 
#                                                     _similarity_top_k
#                                                     )

#     # Get fusion tree filter sort detail engine using _bm25_filter_retriever 
#     # and _vector_retriever (Note: _vector_filter_retriever is not used in this engine)
#     _fusion_tree_filter_sort_detail_engine = get_fusion_tree_keyphrase_filter_sort_detail_engine(
#                                                     _vector_retriever,
#                                                     _vector_docstore,
#                                                     _bm25_filter_retriever,
#                                                     _fusion_top_n,
#                                                     _num_queries,
#                                                     _rerank
#                                                     )   

#     _fusion_tree_keyphrase_sort_detail_tool = QueryEngineTool.from_defaults(
#         name="fusion_keyphrase_tool",
#         query_engine=_fusion_tree_filter_sort_detail_engine,
#         description=_specific_tool_description,
#     )

#     return _fusion_tree_keyphrase_sort_detail_tool


# def get_fusion_tree_keyphrase_sort_detail_tool(
#                                 vector_index: VectorStoreIndex,
#                                 vector_docstore: MongoDocumentStore,
#                                 similarity_top_k_keyphrase: int,
#                                 similarity_top_k_fusion: int,
#                                 fusion_top_n: int,
#                                 query_str: str,
#                                 num_queries: int,
#                                 rerank: BaseNodePostprocessor,
#                                 tool_description: str,
#                                 ) -> QueryEngineTool:
#     """
#     Create a QueryEngineTool that uses a fusion tree filter sort detail engine.
#     The engine is built using a vector retriever and a BM25 filter retriever.

#     Args:
#         vector_index (VectorStoreIndex): The vector store index.
#         vector_docstore (MongoDocumentStore): The vector document store.
#         similarity_top_k_keyphrase (int): Number of similar nodes to retrieve for keyphrase.
#         similarity_top_k_fusion (int): Number of similar nodes to retrieve for fusion.
#         fusion_top_n (int): Number of nodes to return from the fusion engine.
#         query_str (str): The query string to use for the BM25 filter retriever.
#         num_queries (int): Number of queries to use for the fusion engine.
#         rerank (BaseNodePostprocessor): The rerank object to use for the fusion engine.
#         tool_description (str): Description for the QueryEngineTool.

#     Returns:
#         QueryEngineTool: A QueryEngineTool that uses the fusion tree filter sort detail engine.
#     """

#     # Retrieve page numbers that contain a keyphrase of the query using BM25
#     # Currently similarity_top_k = 36, use 18 for bm25 retriever to reduce noise
#     # TODO: tune this parameter
#     similarity_top_n_keyphrase = int(similarity_top_k_keyphrase/2)

#     page_numbers = get_page_numbers_from_query_keyphrase(
#         vector_docstore,
#         similarity_top_n_keyphrase,
#         query_str,
#     )

#     # Create vector retriever with metadata filter using page numbers
#     vector_filter_retriever = vector_index.as_retriever(
#         similarity_top_k=similarity_top_k_keyphrase,  # Retrieve more nodes for filtering
#         filters=MetadataFilters.from_dicts(
#             [
#                 {
#                     "key": "source",
#                     "value": page_numbers,
#                     "operator": "in",
#                 }
#             ]
#         ),
#     )

#     # Get BM25 filter retriever to build a fusion engine with metadata filter
#     bm25_filter_retriever = get_bm25_filter_retriever(
#         vector_filter_retriever,
#         query_str,
#         vector_docstore
#     )

#     # Create vector retriever without metadata filter
#     vector_retriever = vector_index.as_retriever(
#         similarity_top_k=similarity_top_k_fusion,
#     )

#     # Retrieve nodes  using the vector retriever and the query
#     scored_nodes = vector_retriever.retrieve(query_str)

#     # Extract TextNodes from NodeWithScore objects
#     text_nodes = [scored_node.node for scored_node in scored_nodes]

#     print(f"Text nodes in keyphrase vector index length is: {len(text_nodes)}")
#     for i, n in enumerate(text_nodes):
#         print(f"Item {i+1} of the text nodes in keyphrase vector index is page: {n.metadata['source']}")

#     # Get fusion tree filter sort detail engine using vector_retriever and bm25_filter_retriever
#     fusion_tree_filter_sort_detail_engine = get_fusion_tree_keyphrase_filter_sort_detail_engine(
#         vector_retriever,
#         vector_docstore,
#         bm25_filter_retriever,
#         fusion_top_n,
#         num_queries,
#         rerank,
#     )

#     fusion_tree_keyphrase_sort_detail_tool = QueryEngineTool.from_defaults(
#         name="fusion_keyphrase_tool",
#         query_engine=fusion_tree_filter_sort_detail_engine,
#         description=tool_description,
#     )

#     return fusion_tree_keyphrase_sort_detail_tool



def get_fusion_tree_keyphrase_sort_detail_tool_simple(
                                vector_index: VectorStoreIndex,
                                vector_docstore: MongoDocumentStore,
                                similarity_top_k_fusion: int,
                                fusion_top_n: int,
                                query_str: str,
                                num_queries: int,
                                rerank: BaseNodePostprocessor,
                                tool_description: str,
                                ) -> QueryEngineTool:
    """
    Create a QueryEngineTool that uses a fusion tree filter sort detail engine.
    The engine is built using a vector retriever and a BM25 filter retriever.

    Args:
        vector_index (VectorStoreIndex): The vector store index.
        vector_docstore (MongoDocumentStore): The vector document store.
        similarity_top_k_keyphrase (int): Number of similar nodes to retrieve for keyphrase.
        similarity_top_k_fusion (int): Number of similar nodes to retrieve for fusion.
        fusion_top_n (int): Number of nodes to return from the fusion engine.
        query_str (str): The query string to use for the BM25 filter retriever.
        num_queries (int): Number of queries to use for the fusion engine.
        rerank (BaseNodePostprocessor): The rerank object to use for the fusion engine.
        tool_description (str): Description for the QueryEngineTool.

    Returns:
        QueryEngineTool: A QueryEngineTool that uses the fusion tree filter sort detail engine.
    """

    # Retrieve page numbers that contain a keyphrase of the query using BM25
    # Currently similarity_top_k = 36, use 18 for bm25 retriever to reduce noise
    # TODO: tune this parameter
    # similarity_top_n_keyphrase = int(similarity_top_k_keyphrase/2)

    text_nodes = get_text_nodes_from_query_keyphrase(
        vector_docstore,
        similarity_top_k_fusion,
        query_str,
    )

    # Get BM25 keyphrase retriever to build a fusion engine
    bm25_keyphrase_retriever= BM25Retriever.from_defaults(
    similarity_top_k=similarity_top_k_fusion,
    # similarity_top_k=len(text_nodes),
    nodes=text_nodes,
)

    # Create vector retriever without metadata filter
    vector_retriever = vector_index.as_retriever(
        similarity_top_k=similarity_top_k_fusion,
    )

    # Retrieve nodes  using the vector retriever and the query
    scored_nodes = vector_retriever.retrieve(query_str)

    # Extract TextNodes from NodeWithScore objects
    text_nodes = [scored_node.node for scored_node in scored_nodes]

    print(f"Text nodes in keyphrase vector index length is: {len(text_nodes)}")
    for i, n in enumerate(text_nodes):
        print(f"Item {i+1} of the text nodes in keyphrase vector index is page: {n.metadata['source']}")

    # Get fusion tree filter sort detail engine using vector_retriever and bm25_filter_retriever
    fusion_tree_filter_sort_detail_engine = get_fusion_tree_keyphrase_filter_sort_detail_engine(
        vector_retriever,
        vector_docstore,
        bm25_keyphrase_retriever,
        fusion_top_n,
        num_queries,
        rerank,
    )

    fusion_tree_keyphrase_sort_detail_tool = QueryEngineTool.from_defaults(
        name="fusion_keyphrase_tool",
        query_engine=fusion_tree_filter_sort_detail_engine,
        description=tool_description,
    )

    return fusion_tree_keyphrase_sort_detail_tool


# def get_fusion_tree_keyphrase_sort_detail_tool(
#                                         _vector_index: VectorStoreIndex,
#                                         _vector_docstore: MongoDocumentStore,
#                                         _similarity_top_k_keyphrase: int,
#                                         _similarity_top_k_fusion: int,
#                                         _fusion_top_n: int,
#                                         _query_str: str,
#                                         _num_queries: int,
#                                         _rerank: BaseNodePostprocessor,
#                                         _specific_tool_description: str,
#                                         ) -> QueryEngineTool:
#     """
#     This function creates a QueryEngineTool that uses a fusion tree filter sort detail engine.
#     The engine is built using a vector retriever and a BM25 filter retriever. The BM25 
#     filter retriever is built using the vector filter retriever and a query string. The fusion 
#     tree filter sort detail engine is then created using the vector retriever, the BM25 
#     filter retriever, the fusion top n, the number of queries, and a rerank object.

#     Args:
#         _vector_index (VectorStoreIndex): The vector store index.
#         _similarity_top_k (int): The number of similar nodes to retrieve.
#         _page_numbers (list): The page numbers to filter the nodes by.
#         _fusion_top_n (int): The number of nodes to return from the fusion engine.
#         _query_str (str): The query string to use for the BM25 filter retriever.
#         _num_queries (int): The number of queries to use for the fusion engine.
#         _rerank (SentenceTransformerRerank): The rerank object to use for the fusion engine.

#     Returns:
#         QueryEngineTool: A QueryEngineTool that uses the fusion tree filter sort detail 
#         engine.
#     """

#     # print(f"The query is: {_query_str}\n")
#     # Retrieves page numbers that contain a keyphrase of the query using bm25
#     page_numbers = get_page_numbers_from_query_keyphrase(
#                                                     _vector_docstore, 
#                                                     _similarity_top_k_keyphrase, 
#                                                     _query_str) 
#     # print("\n")
#     # for p in page_numbers:
#     #     print(f"Page number that contains the keyphrase: {p}")
        
#     # Create vector retreiver (against a query) with metadata filter using page numbers
#     _vector_filter_retriever = _vector_index.as_retriever(
#                                     similarity_top_k=_similarity_top_k_fusion,
#                                     filters=MetadataFilters.from_dicts(
#                                         [{
#                                             "key": "source", 
#                                             "value": page_numbers,
#                                             "operator": "in"
#                                         }]
#                                     )
#                                 )

#     # Create vector retreiver (against a query) with metadata filter using page numbers
#     _vector_retriever = _vector_index.as_retriever(
#                                     similarity_top_k=_similarity_top_k_fusion,
#                                 )

#     # Get bm25 filter retriever to build a fusion engine with metadata filter 
#     # (query_str is for getting the nodes first)
#     _bm25_filter_retriever = get_bm25_filter_retriever(
#                                                     _vector_filter_retriever, 
#                                                     _query_str, 
#                                                     _similarity_top_k_fusion
#                                                     )

#     # Get fusion tree filter sort detail engine using _bm25_filter_retriever 
#     # and _vector_retriever (Note: _vector_filter_retriever is not used in this engine)
#     _fusion_tree_filter_sort_detail_engine = get_fusion_tree_keyphrase_filter_sort_detail_engine(
#                                                     _vector_retriever,
#                                                     _vector_docstore,
#                                                     _bm25_filter_retriever,
#                                                     _fusion_top_n,
#                                                     _num_queries,
#                                                     _rerank
#                                                     )   

#     _fusion_tree_keyphrase_sort_detail_tool = QueryEngineTool.from_defaults(
#         name="fusion_keyphrase_tool",
#         query_engine=_fusion_tree_filter_sort_detail_engine,
#         description=_specific_tool_description,
#     )

#     return _fusion_tree_keyphrase_sort_detail_tool


# def get_fusion_accumulate_page_filter_sort_detail_engine(
#                                             _vector_filter_retriever,
#                                             _similarity_top_k_filter: int,
#                                             _fusion_top_n_filter: int,
#                                             _query_str: str,
#                                             _num_queries_filter: int,
#                                             ):

#     # Get bm25 filter retriever to build a fusion engine with metadata filter (query_str is for getting the nodes first)
#     _bm25_filter_retriever = get_bm25_filter_retriever(
#                                                 _vector_filter_retriever, 
#                                                 _query_str, 
#                                                 _similarity_top_k_filter
#                                                 )

#     # Get fusion accumulate filter sort detail engine
#     _fusion_accumulate_filter_sort_detail_engine = get_fusion_accumulate_filter_sort_detail_engine(
#                                                                         _vector_filter_retriever,
#                                                                         _bm25_filter_retriever,
#                                                                         _fusion_top_n_filter,
#                                                                         _num_queries_filter
#                                                                         )

#     return _fusion_accumulate_filter_sort_detail_engine


def get_fusion_tree_page_filter_sort_detail_engine(
    vector_filter_retriever,
    similarity_top_k_filter: int,
    fusion_top_n_filter: int,
    text_nodes: List[TextNode],
    num_queries_filter: int,
    rerank: BaseNodePostprocessor,
    vector_docstore: MongoDocumentStore,
    page_numbers: Optional[List[str]]=None,
    ):
    """
    This function creates a fusion tree page filter sort detail engine.

    Parameters:
    - vector_filter_retriever: The vector filter retriever.
    - similarity_top_k_filter (int): The number of top similar documents to retrieve.
    - fusion_top_n_filter (int): The number of top documents to retrieve after fusion.
    - query_str (str): The query string.
    - num_queries_filter (int): The number of queries for filtering.

    Returns:
    - fusion_tree_filter_sort_detail_engine: The fusion tree page filter sort detail engine.
    """

    print(f"similarity_top_k_filter: {similarity_top_k_filter} ")
    print(f"fusion_top_n_filter: {fusion_top_n_filter} ")

    # # Get bm25 filter retriever to build a fusion engine with metadata filter
    # bm25_filter_retriever = get_bm25_filter_retriever(
    #     vector_filter_retriever,
    #     query_str,
    #     # similarity_top_k_filter
    # )

    # Create BM25 filter retriever using the extracted TextNodes
    bm25_filter_retriever= BM25Retriever.from_defaults(
        similarity_top_k=len(text_nodes),
        nodes=text_nodes,
    )

    # Get fusion tree filter sort detail engine
    fusion_tree_filter_sort_detail_engine = get_fusion_tree_filter_sort_detail_engine(
        vector_filter_retriever,
        bm25_filter_retriever,
        fusion_top_n_filter,
        num_queries_filter,
        rerank,
        vector_docstore,
        page_numbers,
        )

    # # Get fusion tree filter sort detail engine
    # if page_numbers is not None:
    #     fusion_tree_filter_sort_detail_engine = get_fusion_tree_filter_sort_detail_engine(
    #         vector_filter_retriever,
    #         bm25_filter_retriever,
    #         fusion_top_n_filter,
    #         num_queries_filter,
    #         rerank,
    #         vector_docstore,
    #         page_numbers,
    #         )
    # else:
    #     fusion_tree_filter_sort_detail_engine = get_fusion_tree_filter_sort_detail_engine(
    #         vector_filter_retriever,
    #         bm25_filter_retriever,
    #         fusion_top_n_filter,
    #         num_queries_filter,
    #         rerank,
    #         vector_docstore,
    #         )

    return fusion_tree_filter_sort_detail_engine


# def get_fusion_tree_page_filter_sort_detail_engine(
#                                         _vector_filter_retriever,
#                                         _similarity_top_k_filter: int,
#                                         _fusion_top_n_filter: int,
#                                         _query_str: str,
#                                         _num_queries_filter: int,
#                                         ):

#     # Get bm25 filter retriever to build a fusion engine with metadata filter (query_str is for getting the nodes first)
#     _bm25_filter_retriever = get_bm25_filter_retriever(
#                                                 _vector_filter_retriever, 
#                                                 _query_str, 
#                                                 _similarity_top_k_filter
#                                                 )

#     # Get fusion tree filter sort detail engine
#     _fusion_tree_filter_sort_detail_engine = get_fusion_tree_filter_sort_detail_engine(
#                                                                         _vector_filter_retriever,
#                                                                         _bm25_filter_retriever,
#                                                                         _fusion_top_n_filter,
#                                                                         _num_queries_filter
#                                                                         )

#     return _fusion_tree_filter_sort_detail_engine


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


def change_tree_engine_prompt_to_in_detail(
        engine,
        page_numbers: Optional[List[str]] = None,
        ):

    if page_numbers is not None:
        page_number_string = ", ".join(page_number for page_number in page_numbers)
        
        new_summary_tmpl_str = (
        f"Context information from pages {page_number_string} is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the information from multiple sources and NOT PRIOR KNOWLEDGE, "
        "answer the query below.\n"
        "TRY TO INCLUDE AS MANY DETAILS AS POSSIBLE ONLY FROM THE PROVIDED CONTEXT \n"
        "IFORMATION. DO NOT INCLUDE ANYTHING THAT IS NOT IN THE PROVIDED CONTEXT INFORMATION.\n"
        "Query: {query_str}\n"
        "Answer: "
        )
    else:
        new_summary_tmpl_str = (
        "Context information from multiple sources is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the information from multiple sources and NOT PRIOR KNOWLEDGE, "
        "answer the query below.\n"
        "TRY TO INCLUDE AS MANY DETAILS AS POSSIBLE ONLY FROM THE PROVIDED CONTEXT \n"
        "IFORMATION. DO NOT INCLUDE ANYTHING THAT IS NOT IN THE PROVIDED CONTEXT INFORMATION.\n"
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
    "Given the information from multiple sources and NOT PRIOR KNOWLEDGE, "
    "answer the query below.\n"
    "Try to provide a summary IN DETAIL that captures the major story lines, \n"
    "main arguments, supporting evidence, and key conclusions. \n"
    "If asked to create table of contents, create a table of contents that \n"
    " 1. includes the major story lines, \n"
    " 2. main arguments, \n"
    " 3. supporting evidence, and \n"
    " 4. key conclusions. \n"
    "DO NOT INCLUDE ANYTHING THAT IS NOT IN THE PROVIDED CONTEXT INFORMATION.\n"
    "Query: {query_str}\n"
    "Answer: "
    )

    # new_summary_tmpl_str = (
    # "Context information from multiple sources is below.\n"
    # "---------------------\n"
    # "{context_str}\n"
    # "---------------------\n"
    # "Given the information from multiple sources and not prior knowledge, "
    # "answer the query below.\n"
    # "Try to provide a detailed summary that captures the major story line, "
    # "main arguments, supporting evidence, and key conclusions in at least 500 words.\n"
    # "Query: {query_str}\n"
    # "Answer: "
    # )

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