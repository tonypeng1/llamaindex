import logging
import os
import sys

from llama_index.core import (
                        Document,
                        Settings,
                        SimpleDirectoryReader,
                        StorageContext,
                        VectorStoreIndex,
                        )
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import (
                        HierarchicalNodeParser,
                        get_leaf_nodes,
                        get_root_nodes,
                        )
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.vector_stores.milvus import MilvusVectorStore

from pymilvus import connections, db, MilvusClient
from pymongo import MongoClient
import openai

from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.readers.file import PyMuPDFReader

from trulens_eval import Tru


def check_if_milvus_database_exists(uri, db_name) -> bool:
    connections.connect(uri=uri)
    db_names = db.list_database()
    connections.disconnect("default")
    return db_name in db_names

def check_if_milvus_collection_exists(uri, db_name, collect_name) -> bool:
    client = MilvusClient(
        uri=uri,
        db_name=db_name
        )
    # client.load_collection(collection_name=collect_name)
    collect_names = client.list_collections()
    client.close()
    return collect_name in collect_names


def check_if_mongo_database_exists(uri, db_name) -> bool:
    client = MongoClient(uri)
    db_names = client.list_database_names()
    client.close()
    return db_name in db_names


def check_if_mongo_namespace_exists(uri, db_name, namespace) -> bool:
    client = MongoClient(uri)
    db = client[db_name]
    collection_names = db.list_collection_names()
    client.close()
    return namespace + "/data" in collection_names  # Choose from 3 in the list

def create_database_milvus(uri, db_name):
    connections.connect(uri=uri)
    db.create_database(db_name)
    connections.disconnect("default")

def milvus_collection_item_count(uri, db_name, collect_name) -> int:
    client = MilvusClient(
        uri=uri,
        db_name=db_name
        )
    client.load_collection(collection_name=collect_name)
    element_count = client.query(
        collection_name=collection_name,
        output_fields=["count(*)"],
        )
    client.close()
    return element_count[0]['count(*)']

def print_retreived_nodes(retrieve):
    # Loop through each NodeWithScore in the retreived nodes
    for (i, node_with_score) in enumerate(retrieve):
        node = node_with_score.node  # The TextNode object
        score = node_with_score.score  # The similarity score
        chunk_id = node.id_  # The chunk ID

        # Extract the relevant metadata from the node
        file_name = node.metadata.get("file_name", "Unknown")
        file_path = node.metadata.get("file_path", "Unknown")

        # Extract the text content from the node
        text_content = node.text if node.text else "No content available"

        # Print the results in a user-friendly format
        print(f"\n\nItem number: {i+1}")
        print(f"Score: {score}")
        # print(f"File Name: {file_name}")
        # print(f"File Path: {file_path}")
        print(f"Id: {chunk_id}")
        print("\nExtracted Content:")
        print(text_content)
        # print("\n" + "=" * 40 + " End of Result " + "=" * 40 + "\n")
        # print("\n")


def load_document_pdf(doc_link):
    loader = PyMuPDFReader()
    docs0 = loader.load(file_path=Path(doc_link))
    docs = Document(text="\n\n".join([doc.text for doc in docs0]))
    # print(type(documents), "\n")
    # print(len(documents), "\n")
    # print(type(documents[0]))
    # print(documents[0])
    return docs

def get_notes_from_document_automerge(docs, sizes=[2048, 512, 1]):
    # create the hierarchical node parser w/ default settings
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=sizes
    )
    nodes = node_parser.get_nodes_from_documents([docs])
    leaf_nodes = get_leaf_nodes(nodes)
    return nodes, leaf_nodes


def check_if_milvus_database_collection_exist(db_name, col_name):
    save_ind = True
    if check_if_milvus_database_exists(uri_milvus, db_name):
        if check_if_milvus_collection_exists(uri_milvus, db_name, col_name):
            num_count = milvus_collection_item_count(uri_milvus, database_name, collection_name)
            if num_count > 0:  # account for the case of 0 item in the collection
                save_ind = False
    else:
        create_database_milvus(uri_milvus, database_name)
    return save_ind

def check_if_mongo_database_namespace_exist(db_name, col_name):
    add_doc = True
    if check_if_mongo_database_exists(uri_mongo, db_name):
        if check_if_mongo_namespace_exists(uri_mongo, db_name, col_name):
            add_doc = False
    return add_doc


def build_automerge_index_and_docstore(
    art_link,
    save_ind,
    add_doc,
    chunck_size,
    ):

    if save_ind or add_doc:  # Only load and parse document if either index or docstore not saved.
        document = load_document_pdf(art_link)
        (nodes, leaf_n) = get_notes_from_document_automerge(document, chunck_size)

    if save_ind == True:
        # Create and save index (embedding) to Milvus database
        base_ind = VectorStoreIndex(
            nodes=leaf_n,
            storage_context=storage_context,
            )
    else:
        # load from Milvus database
        base_ind = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
            )

    if add_doc == True:
        # Save document nodes to Mongodb docstore at the server
        storage_context.docstore.add_documents(nodes)

    return base_ind


def get_automerge_retriever(
        _base_index,
        _similarity_top_k,
        ):
    # Create the retriever
    base_retrieve = _base_index.as_retriever(similarity_top_k=_similarity_top_k)

    retrieve = AutoMergingRetriever(
        vector_retriever=base_retrieve, 
        # storage_context=automerging_index.storage_context,  # This does not work, results in dim mismatch. 
        storage_context=storage_context, 
        verbose=True
        )
    
    return base_retrieve, retrieve

def get_automerge_query_engine(
    base_retrieve,
    retrieve,
    rank_model,
    rank_top_n,
    ):
    base_query_engi = RetrieverQueryEngine.from_args(
        retriever=base_retrieve
        )
    query_engi = RetrieverQueryEngine.from_args(
        retriever=retrieve
        )
    rerank = SentenceTransformerRerank(
        top_n=rank_top_n,
        model=rank_model,
        )
    rerank_engine = RetrieverQueryEngine.from_args(
        retriever=retrieve, 
        node_postprocessors=[rerank],
        )
    return base_query_engi, query_engi, rerank_engine

    
# # Set up logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set OpenAI API key, LLM, and embedding model
openai.api_key = os.environ['OPENAI_API_KEY']
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.llm = llm
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# embed_model_dim = 384  # for bge-small-en-v1.5
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
embed_model_dim = 1536  # for text-embedding-3-small

uri_milvus = "http://localhost:19530"
uri_mongo = "mongodb://localhost:27017/"

# Create database and collection names

# article_dictory = "paul_graham"
# article_name = "paul_graham_essay.pdf"

article_dictory = "andrew"
article_name = "eBook-How-to-Build-a-Career-in-AI.pdf"

article_link = "./data/" + article_dictory + "/" + article_name

chuck_method = "automerge"
leaf = 128
parent_1 = 512
parent_2 = 2048
# leaf = 256
# parent_1 = 1024
# parent_2 = 4096
automerge_chuck_size = [leaf, parent_1, parent_2]
embed_model = "openai_embedding_3_small"
# embed_model = "huggingface_embedding_bge_small"

database_name = article_dictory + "_" + chuck_method
collection_name = embed_model + "_size_" + str(parent_2) + "_" + str(parent_1) + "_" + str(leaf)


# Check if index and docstore have already been saved to Milvus and MongoDB.

# In Milvus database, if the specific collection exists , do not save the index.
# Otherwise, create one.
save_index = check_if_milvus_database_collection_exist(database_name, collection_name)

# In MongoDB, if the specific namespace exists, do not add document nodes to MongoDB.
add_document = check_if_mongo_database_namespace_exist(database_name, collection_name)


# Initiate vector store, docstore, and storage context.

# Initiate vector store (a new empty collection will be created in Milvus server)
vector_store = MilvusVectorStore(
    uri=uri_milvus,
    db_name=database_name,
    collection_name=collection_name,
    dim=embed_model_dim,  # dim of HuggingFace "BAAI/bge-small-en-v1.5" embedding model
    )

# Initiate MongoDB docstore (Not yet save to MongoDB server)
docstore = MongoDocumentStore.from_uri(
    uri=uri_mongo,
    db_name=database_name,
    namespace=collection_name
    )

# Initiate storage context: use Milvus as vector store and Mongo as docstore 
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    docstore=docstore
    )
# for i in list(storage_context.docstore.get_all_ref_doc_info().keys()):
#     print(i)
# print(storage_context.docstore.get_node(leaf_nodes[0].node_id))


# Get base index
base_index = build_automerge_index_and_docstore(
    article_link,
    save_index,
    add_document,
    automerge_chuck_size,
)

# Get retrievers and query engines
similarity_top_k = 12
rerank_model = "BAAI/bge-reranker-base"
rerank_top_n =6

base_retriever, retriever = get_automerge_retriever(base_index, similarity_top_k)
base_query_engine, query_engine, rerank_query_engine = get_automerge_query_engine(base_retriever,
                                                                                  retriever,
                                                                                  rerank_model,
                                                                                  rerank_top_n)

query_str = "What are the keys to building a career in AI?"
# query_str = "What happened in New York?"
# query_str = "What happened at Interleafe and Viaweb?"
# query_str = "What is the importance of networking in AI?"
# query_str = (
#     "What could be the potential outcomes of adjusting the amount of safety"
#     " data used in the RLHF stage?"
# )

# Print retrieved nodes
vector_store.client.load_collection(collection_name=collection_name)

base_nodes_retrieved = base_retriever.retrieve(query_str)
print_retreived_nodes(base_nodes_retrieved)

nodes_retrieved = retriever.retrieve(query_str)
print_retreived_nodes(nodes_retrieved)

# Prints responses 
base_response = base_query_engine.query(query_str)
print("\nBASE:\n" + str(base_response))

response = query_engine.query(query_str)
print("\nAUTO-MERGE:\n" + str(response))

rerank_response = rerank_query_engine.query(query_str)
print("\nRE-RANK:\n" + str(rerank_response))

# print(rerank_response.get_formatted_sources(length=2000))



vector_store.client.release_collection(collection_name=collection_name)
vector_store.client.close()  # Need to do this (otherwise Milvus container will hault when closing)
# del docstore  # MongoDB may frnot need to be manually closed.






