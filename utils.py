from typing import List, Optional, Union, Dict, Any
import re

from keybert import KeyBERT
from llama_index.core import (
                        PromptTemplate,
                        QueryBundle,
                        StorageContext,
                        SummaryIndex,
                        VectorStoreIndex
                        )
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.indices.postprocessor import (
                        PrevNextNodePostprocessor,
                        )
from llama_index.core.postprocessor.node import get_forward_nodes, get_backward_nodes
from llama_index.core.retrievers import (
                        QueryFusionRetriever, 
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


class SafePrevNextNodePostprocessor(PrevNextNodePostprocessor):
    """
    A custom PrevNextNodePostprocessor that handles missing nodes gracefully.
    """
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        all_nodes = {}
        for node in nodes:
            all_nodes[node.node.node_id] = node

            if self.mode in ["next", "both"]:
                try:
                    forward_nodes = get_forward_nodes(node, self.num_nodes, self.docstore)
                    all_nodes.update(forward_nodes)
                except Exception as e:
                    print(f"Warning: Failed to get next nodes for {node.node.node_id}: {e}")

            if self.mode in ["prev", "both"]:
                try:
                    backward_nodes = get_backward_nodes(node, self.num_nodes, self.docstore)
                    all_nodes.update(backward_nodes)
                except Exception as e:
                    print(f"Warning: Failed to get prev nodes for {node.node.node_id}: {e}")
        
        return list(all_nodes.values())


class PageSortNodePostprocessor(BaseNodePostprocessor):
    """
    A custom node postprocessor that sorts nodes based on the page number and the order they appear in a document.

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
        (IMPORTANT: ALTHOUGH QUERY_BUNDLE IS NOT USED IN THIS METHOD, IT IS REQUIRED FOR THE 
        INTERFACE. IF REMOVED WILL CAUSE AN ERROR OF TOO MANY POSITIONAL ARGUMENTS.)
        
        Args:
            nodes (List[NodeWithScore]): A list of nodes to be sorted.

        Returns:
            List[NodeWithScore]: A list of nodes sorted based on the page they appear on and the order they appear 
            in that page.
        """

        try:
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
        except Exception as e:
            print(f"Error in PageSortNodePostprocessor: {e}")
            return nodes


class PrintNodesPostprocessor(BaseNodePostprocessor):
    """
    A custom node postprocessor that prints the content of the nodes.
    """
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        print("\n" + "="*80)
        print(f"🔍 INSPECTING {len(nodes)} NODES BEFORE LLM SYNTHESIS")
        print("="*80)
        
        for i, node in enumerate(nodes):
            print(f"\n📄 NODE {i+1}/{len(nodes)} (Score: {node.score})")
            print(f"   ID: {node.node.node_id}")
            print(f"   Metadata: {node.node.metadata}")
            print("-" * 40)
            print("   CONTENT:")
            print(node.node.get_content())
            print("-" * 40)
            
        print("\n" + "="*80 + "\n")
        return nodes


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
        article_name: str,
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
    article_name = article_name.split(".")[0]  # Remove the file extension
    database_name = f"{article_directory}_{article_name}_{chunk_method}"

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
        keyphrase_ngram_range=(1, 3),  # Allow 1-3 word phrases for flexibility
        top_n=5,
    )

    # Create a set to store unique keywords
    keywords_set = set()
    for k in keywords:
        if k[0]:  # Check if keyphrase is not empty
            keywords_set.update(k[0].split())

    # Concatenate the keywords into a single string
    final_keywords = " ".join(keywords_set)

    # Fallback to original query if no keywords extracted
    if not final_keywords or final_keywords.strip() == ".":
        print(f"   Warning: No valid keyphrases extracted, using original query")
        return query_str

    # Return the final keyphrase string
    return final_keywords


def get_fusion_tree_filter_sort_detail_engine(
    vector_filter_retriever,
    bm25_filter_retriever: BM25Retriever,
    fusion_top_n: int,
    num_queries: int,
    rerank: BaseNodePostprocessor,
    vector_docstore: MongoDocumentStore,
    page_numbers: Optional[List[str]]=None,
    num_nodes: int = 0,
):
    """
    Build a fusion tree filter sort detail engine using vector filter retriever and bm25 filter retriever.

    Args:
        vector_filter_retriever: Vector filter retriever object.
        bm25_filter_retriever (BM25Retriever): BM25 filter retriever object.
        fusion_top_n (int): The number of top documents to consider for fusion.
        num_queries (int): The number of queries to generate for fusion.
        rerank (BaseNodePostprocessor): The rerank object to use.
        vector_docstore (MongoDocumentStore): The vector document store.
        page_numbers (Optional[List[str]]): Optional list of page numbers to filter.
        num_nodes (int): Number of neighboring nodes to retrieve (default: 0).

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

    PrevNext = SafePrevNextNodePostprocessor(
                                docstore=vector_docstore,
                                num_nodes=num_nodes,  # retrieve n nodes before and n after
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

    return fusion_tree_filter_sort_detail_engine


def get_fusion_tree_keyphrase_filter_sort_detail_engine(
                                                vector_retriever,
                                                vector_docstore: MongoDocumentStore,
                                                bm25_retriever: BM25Retriever,
                                                fusion_top_n: int,
                                                num_queries: int,
                                                rerank: ColbertRerank = None,
                                                num_nodes: int = 0,
                                                ):
    """
    This function creates a fusion filter retriever and engine that combines results from a vector retriever and a BM25 retriever.
    It also applies a PrevNext node postprocessor to the results, and a PageSort node postprocessor.
    If a rerank function is provided, it will be applied as well.

    Parameters:
    vector_retriever (object): The vector retriever to be used in the fusion.
    vector_docstore (MongoDocumentStore): The document store to be used in the fusion.
    bm25_retriever (BM25Retriever): The BM25 retriever to be used in the fusion.
    fusion_top_n (int): The number of top results to consider from each retriever in the fusion.
    num_queries (int): The number of queries to generate. Set to 1 to disable query generation.
    rerank (ColbertRerank, optional): A rerank function to be applied to the results. Defaults to None.
    num_nodes (int): Number of neighboring nodes to retrieve (default: 0).

    Returns:
    fusion_tree_filter_sort_detail_engine (RetrieverQueryEngine): The fusion tree filter sort detail engine created using the provided parameters.
    """

    # Create fusion filter retreiver and engine
    fusion_filter_retriever = QueryFusionRetriever(
                                retrievers=[
                                        vector_retriever, 
                                        bm25_retriever
                                        ],
                                similarity_top_k=fusion_top_n,
                                num_queries=num_queries,  # set this to 1 to disable query generation
                                mode="relative_score",
                                retriever_weights=[0.5, 0.5],
                                use_async=True,
                                verbose=True,
                                )

    PrevNext = SafePrevNextNodePostprocessor(
                                    docstore=vector_docstore,
                                    num_nodes=num_nodes,  # retrieve n nodes before and n after
                                    mode="both",
                                    verbose=True,
                                    )  # can only retrieve the two nodes on one page

    node_postprocessors = [
                        PrevNext,
                        PageSortNodePostprocessor(),
                        # PrintNodesPostprocessor(),
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


def extract_entities_from_query(
        query_str: str,
        llm=None
        ) -> Dict[str, List[str]]:
    """
    Extract named entities from a query string using EntityExtractor.
    
    This function uses the same EntityExtractor model used for document processing
    to identify person names, organizations, locations, instruments, diseases, and other entities mentioned in the user query.
    
    Parameters:
    query_str (str): The user query string
    llm: The LLM instance (not used, kept for API compatibility)
    
    Returns:
    Dict[str, List[str]]: Dictionary with entity types as keys and lists of entities as values
                         Example: {'PER': ['Paul Graham'], 'ORG': ['Y Combinator']}
    """
    from llama_index.extractors.entity import EntityExtractor
    from llama_index.core.schema import Document
    
    # Initialize EntityExtractor with the same model used for document processing
    entity_extractor = EntityExtractor(
        model_name="lxyuan/span-marker-bert-base-multilingual-cased-multinerd",
        prediction_threshold=0.5,
        label_entities=True,
        device="mps",  # Use "cpu" if not on Apple Silicon
    )
    
    # Create a temporary document from the query string
    temp_doc = Document(text=query_str)
    
    # Extract entities
    try:
        # EntityExtractor expects a list of nodes
        from llama_index.core.node_parser import SentenceSplitter
        node_parser = SentenceSplitter(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents([temp_doc])
        
        # Process nodes through entity extractor
        processed_nodes = entity_extractor.process_nodes(nodes)
        
        # Collect entities from all processed nodes
        # We initialize with empty lists for all supported types to ensure consistent structure
        # The keys correspond to the MultiNERD dataset labels
        entities = {
            'PER': [], 'ORG': [], 'LOC': [], 'ANIM': [], 'BIO': [], 'CEL': [], 
            'DIS': [], 'EVE': [], 'FOOD': [], 'INST': [], 'MEDIA': [], 
            'PLANT': [], 'MYTH': [], 'TIME': [], 'VEHI': []
        }
        
        for node in processed_nodes:
            metadata = node.metadata
            
            # EntityExtractor stores entities in metadata with keys like 'PER', 'ORG', 'LOC'
            # when label_entities=False, or 'persons', 'organizations', 'locations' when label_entities=True
            # Map both formats to standardized keys. 
            # This mapping covers the labels from the MultiNERD dataset which the model is trained on.
            key_mappings = {
                'PER': ['PER', 'persons'],
                'ORG': ['ORG', 'organizations'],
                'LOC': ['LOC', 'locations'],
                'ANIM': ['ANIM', 'animals'],
                'BIO': ['BIO', 'biologicals'],
                'CEL': ['CEL', 'celestial_bodies'],
                'DIS': ['DIS', 'diseases'],
                'EVE': ['EVE', 'events'],
                'FOOD': ['FOOD', 'foods'],
                'INST': ['INST', 'instruments'],
                'MEDIA': ['MEDIA', 'media'],
                'PLANT': ['PLANT', 'plants'],
                'MYTH': ['MYTH', 'myths'],
                'TIME': ['TIME', 'times'],
                'VEHI': ['VEHI', 'vehicles'],
            }
            
            for standard_key, possible_keys in key_mappings.items():
                for entity_type in possible_keys:
                    if entity_type in metadata and metadata[entity_type]:
                        # metadata[entity_type] is a list of entities
                        for entity in metadata[entity_type]:
                            if entity not in entities[standard_key]:
                                entities[standard_key].append(entity)
        
        # Remove empty lists
        entities = {k: v for k, v in entities.items() if v}
        
        # Print extracted entities
        if entities:
            print(f"\n📌 Entities extracted from query:")
            for entity_type, entity_list in entities.items():
                print(f"   {entity_type}: {entity_list}")
        else:
            print(f"\n📌 No entities extracted from query\n")
        
        return entities
        
    except Exception as e:
        print(f"Warning: Entity extraction from query failed: {e}")
        print(f"Returning empty entity dict")
        return {}


def create_entity_metadata_filters(
        entities: Dict[str, List[str]],
        metadata_option: str,
        ) -> Optional[MetadataFilters]:
    """
    Create MetadataFilters for entity-based filtering.
    
    Parameters:
    entities (Dict[str, List[str]]): Dictionary of entities by type (PER, ORG, LOC, INST, DIS, etc.)
    metadata_option (str): The metadata extraction option used ('entity', 'langextract', or 'both')
    
    Returns:
    Optional[MetadataFilters]: MetadataFilters object or None if no entities
    """
    if not entities:
        return None
    
    filters = []
    
    # Map standardized keys (PER, ORG, LOC) to actual database keys (persons, organizations, locations)
    # This mapping should match the output format of the EntityExtractor when label_entities=True
    key_mapping = {
        'PER': 'persons',
        'ORG': 'organizations',
        'LOC': 'locations',
        'ANIM': 'animals',
        'BIO': 'biologicals',
        'CEL': 'celestial_bodies',
        'DIS': 'diseases',
        'EVE': 'events',
        'FOOD': 'foods',
        'INST': 'instruments',
        'MEDIA': 'media',
        'PLANT': 'plants',
        'MYTH': 'myths',
        'TIME': 'times',
        'VEHI': 'vehicles',
    }
    
    # For EntityExtractor metadata (persons, organizations, locations fields in database)
    if metadata_option in ["entity", "both"]:
        for entity_type, entity_list in entities.items():
            # Convert PER/ORG/LOC to lowercase database keys
            db_key = key_mapping.get(entity_type, entity_type)
            for entity in entity_list:
                # For list fields, checks if value is in list
                filters.append({
                    "key": db_key,
                    "value": entity,
                    "operator": "=="
                })
    
    # For LangExtract metadata (entity_names field)
    if metadata_option in ["langextract", "both"]:
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)
        
        if all_entities:
            for entity in all_entities:
                filters.append({
                    "key": "entity_names",
                    "value": entity,
                    "operator": "=="
                })
    
    if not filters:
        return None
    
    # Use OR condition - retrieve nodes that mention ANY of the entities
    return MetadataFilters.from_dicts(filters, condition="or")


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

    print(f"\nThe query in keyphrase tool is: {query_str}")
    print(f"\n📌The keyphrase is: {query_keyphrase}\n")

    # Get page numbers containing the keyphrase using bm25 model, sorted in ascending order
    bm25_score_nodes = bm25_retriever.retrieve(query_keyphrase)
    bm25_text_nodes = [node.node for node in bm25_score_nodes ]  # get TextNode from ScoredNode

    return bm25_text_nodes


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
                    using the provided storage_context.
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


def get_fusion_tree_keyphrase_sort_detail_tool_simple(
                                vector_index: VectorStoreIndex,
                                vector_docstore: MongoDocumentStore,
                                similarity_top_k_fusion: int,
                                fusion_top_n: int,
                                query_str: str,
                                num_queries: int,
                                rerank: BaseNodePostprocessor,
                                tool_description: str,
                                enable_entity_filtering: bool = False,
                                metadata_option: Optional[str] = None,
                                llm = None,
                                num_nodes: int = 0,
                                ) -> QueryEngineTool:
    """
    Create a QueryEngineTool that uses a fusion tree filter sort detail engine.
    The engine is built using a vector retriever and a BM25 filter retriever,
    with optional entity-based metadata filtering.

    Args:
        vector_index (VectorStoreIndex): The vector store index.
        vector_docstore (MongoDocumentStore): The vector document store.
        similarity_top_k_fusion (int): Number of similar nodes to retrieve for fusion.
        fusion_top_n (int): Number of nodes to return from the fusion engine.
        query_str (str): The query string to use for the BM25 filter retriever.
        num_queries (int): Number of queries to use for the fusion engine.
        rerank (BaseNodePostprocessor): The rerank object to use for the fusion engine.
        tool_description (str): Description for the QueryEngineTool.
        enable_entity_filtering (bool): Whether to enable entity-based filtering (default: False)
        metadata_option (str): Metadata extraction option used ('entity', 'langextract', or 'both')
        llm: The LLM instance for entity extraction (optional)

    Returns:
        QueryEngineTool: A QueryEngineTool that uses the fusion tree filter sort detail engine.
    """

    # STEP 1: Entity-based filtering (if enabled)
    entity_filters = None
    extracted_entities = {}
    
    if enable_entity_filtering and metadata_option in ["entity", "langextract", "both"]:
        # Extract entities from the query
        extracted_entities = extract_entities_from_query(query_str, llm)
        
        if extracted_entities:
            print(f"\n🔍 Entity Filtering Enabled")
            print(f"   Extracted entities from query: {extracted_entities}")
            
            # Create metadata filters for entity-based filtering
            entity_filters = create_entity_metadata_filters(extracted_entities, metadata_option)
            
            if entity_filters:
                print(f"   Applying entity filters to retrieval...\n")
    
    # STEP 2: Keyphrase extraction for BM25
    text_nodes = get_text_nodes_from_query_keyphrase(
        vector_docstore,
        similarity_top_k_fusion,
        query_str,
    )

    # Get BM25 keyphrase retriever to build a fusion engine
    bm25_keyphrase_retriever= BM25Retriever.from_defaults(
        similarity_top_k=similarity_top_k_fusion,
        nodes=text_nodes,
        )

    # STEP 3: Create vector retriever (with entity filter if applicable)
    # DEBUG: print entity_filters for troubleshooting before creating retriever
    print("\nDEBUG: entity_filters (raw):", entity_filters)
    if entity_filters:
        try:
            print("DEBUG: entity_filters details:")
            # MetadataFilters typically exposes `.filters` and `.condition`
            if hasattr(entity_filters, 'filters'):
                print(f"  num_filters: {len(entity_filters.filters)}")
                for f in entity_filters.filters:
                    try:
                        print(f"    key={f.key}, value={f.value}, operator={f.operator}")
                    except Exception:
                        print(f"    filter item: {f}")
            else:
                print(f"  entity_filters has no .filters attribute: {entity_filters}")
            if hasattr(entity_filters, 'condition'):
                print(f"  condition: {entity_filters.condition}")
        except Exception as e:
            print(f"DEBUG: failed to introspect entity_filters: {e}")
    if entity_filters:
        # Entity-filtered vector retriever
        vector_retriever = vector_index.as_retriever(
            similarity_top_k=similarity_top_k_fusion,
            filters=entity_filters
        )
        print(f"✓ Vector retriever created WITH entity filtering")
    else:
        # Standard vector retriever without entity filter
        vector_retriever = vector_index.as_retriever(
            similarity_top_k=similarity_top_k_fusion,
        )
        if enable_entity_filtering:
            print(f"\n⚠️  Entity filtering enabled but no entities found in query")
            print(f"   Using standard retrieval without entity filters\n")

    # Retrieve nodes using the vector retriever and the query
    scored_nodes = vector_retriever.retrieve(query_str)

    # Extract TextNodes from NodeWithScore objects
    text_nodes = [scored_node.node for scored_node in scored_nodes]

    print(f"\nText nodes in keyphrase vector index length is: {len(text_nodes)}")
    if entity_filters and extracted_entities:
        print(f"   (Filtered by entities: {', '.join([e for elist in extracted_entities.values() for e in elist])})")
    # for i, n in enumerate(text_nodes):
    #     print(f"Item {i+1} of the text nodes in keyphrase vector index is page: {n.metadata['source']}")

    # Get fusion tree filter sort detail engine using vector_retriever and bm25_filter_retriever
    fusion_tree_filter_sort_detail_engine = get_fusion_tree_keyphrase_filter_sort_detail_engine(
        vector_retriever,
        vector_docstore,
        bm25_keyphrase_retriever,
        fusion_top_n,
        num_queries,
        rerank,
        num_nodes,
    )

    fusion_tree_keyphrase_sort_detail_tool = QueryEngineTool.from_defaults(
        name="fusion_keyphrase_tool",
        query_engine=fusion_tree_filter_sort_detail_engine,
        description=tool_description,
    )

    return fusion_tree_keyphrase_sort_detail_tool


def get_fusion_tree_page_filter_sort_detail_engine(
    vector_filter_retriever,
    fusion_top_n_filter: int,
    text_nodes: List[TextNode],
    num_queries_filter: int,
    rerank: BaseNodePostprocessor,
    vector_docstore: MongoDocumentStore,
    page_numbers: Optional[List[str]]=None,
    ):

    print(f"fusion_top_n_filter: {fusion_top_n_filter} ")

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

    return fusion_tree_filter_sort_detail_engine


def change_tree_engine_prompt_to_in_detail(
        engine,
        page_numbers: Optional[List[str]] = None,
        ):
    """
    This function modifies the engine's prompt to include more detailed responses based on the 
    context information.

    Parameters:
    engine (object): The engine object which contains the prompt to be modified.
    page_numbers (Optional[List[str]], optional): A list of page numbers from which the context 
    information is sourced. Defaults to None.

    Returns:
    engine (object): The modified engine object with the updated prompt.
    """

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


def change_summary_engine_prompt_to_in_detail(engine):
    """
    This function modifies the summary engine prompt to provide a more detailed summary.

    Parameters:
    engine (object): The engine object that needs to be modified.

    Returns:
    engine (object): The modified engine object with the new summary template.
    """

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

    new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    engine.update_prompts(
                    {"response_synthesizer:summary_template": 
                                            new_summary_tmpl}
                                            )
    return engine
