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

def automerge_create_or_load_index(save_ind):
    if save_ind == True:
        # Create and save index (embedding) to Milvus database
        base_ind = VectorStoreIndex(
            nodes=leaf_nodes, 
            storage_context=storage_context, 
            )
    else:
        # load from Milvus database
        base_ind = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )

    return(base_ind)

# # Set up logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set OpenAI API key, LLM, and embedding model
openai.api_key = os.environ['OPENAI_API_KEY']
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

uri_milvus = "http://localhost:19530"
uri_mongo = "mongodb://localhost:27017/"

# datbase_name: article + RAG approach, collection_name: configuration
# database_name = "andrew_ai_article_automerge"
# collection_name = "three_layer_2048_512_128"

# article_link = "./data/andrew/eBook-How-to-Build-a-Career-in-AI.pdf" 
# article_link = "./data/paul_graham/paul_graham_essay.pdf" 

# article_link = "./data/andrew/"
# article_link = "./data/llama/"
# article_link = "./data/llama_2.pdf"  # not working
article_dictory = "paul_graham"
article_name = "paul_graham_essay.pdf"
article_link = "./data/" + article_dictory + "/" + article_name
chuck_method = "automerge"
automerge_chuck_size = [2048, 512, 128]

database_name = article_dictory + "_" + chuck_method
collection_name = "size_2048_512_128"

# database_name = "llama2_article_automerge"
# collection_name = "three_layer_2048_512_128"

# In Milvus database, if the specific collection exists , do not save the index.
# If the database does not exist, create one.
save_index = check_if_milvus_database_collection_exist(database_name, collection_name)

# save_index = True
# if check_if_milvus_database_exists(uri_milvus, database_name):
#     if check_if_milvus_collection_exists(uri_milvus, database_name, collection_name):
#         num_count = milvus_collection_item_count(uri_milvus, database_name, collection_name)
#         if num_count > 0:  # account for the case of 0 item in the collection
#             save_index = False
# else:
#     create_database_milvus(uri_milvus, database_name)

# In MongoDB, if the specific namespace exists, do not add document nodes to MongoDB.
add_document = check_if_mongo_database_namespace_exist(database_name, collection_name)

# add_document = True
# if check_if_mongo_database_exists(uri_mongo, database_name):
#     if check_if_mongo_namespace_exists(uri_mongo, database_name, collection_name):
#         add_document = False


if save_index or add_document:  # Only load and parse document if new
    document = load_document_pdf(article_link)
    (nodes, leaf_nodes) = get_notes_from_document_automerge(document, automerge_chuck_size)


# Initiate vector store (a new empty collection created in Milvus server)
vector_store = MilvusVectorStore(
    uri=uri_milvus,
    db_name=database_name,
    collection_name=collection_name,
    dim=384,  # dim of HuggingFace "BAAI/bge-small-en-v1.5" embedding model
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

# if save_index == True:
#     # Create and save index (embedding) to Milvus database
#     base_index = VectorStoreIndex(
#         nodes=leaf_nodes, 
#         storage_context=storage_context, 
#         )
# else:
#     # load from Milvus database
#     base_index = VectorStoreIndex.from_vector_store(
#         vector_store=vector_store
#     )

# if add_document == True:
#     # Save document nodes to Mongodb docstore at the server
#     storage_context.docstore.add_documents(nodes)

if add_document == True:
    # Save document nodes to Mongodb docstore at the server
    storage_context.docstore.add_documents(nodes)

base_index = automerge_create_or_load_index(save_index)

# Create the retriever
base_retriever = base_index.as_retriever(
    similarity_top_k=12
)

retriever = AutoMergingRetriever(
    vector_retriever=base_retriever, 
    # storage_context=automerging_index.storage_context,  # This does not work, results in dim mismatch. 
    storage_context=storage_context, 
    verbose=True
)

query_str = "What happened at Interleafe and Viaweb?"
# query_str = "What is the importance of networking in AI?"
# query_str = (
#     "What could be the potential outcomes of adjusting the amount of safety"
#     " data used in the RLHF stage?"
# )

vector_store.client.load_collection(collection_name=collection_name)
base_nodes_retrieved = base_retriever.retrieve(query_str)
print_retreived_nodes(base_nodes_retrieved)

nodes_retrieved = retriever.retrieve(query_str)
print_retreived_nodes(nodes_retrieved)

# len(retrieved_base)
# len(retrieved)

base_query_engine = RetrieverQueryEngine.from_args(
    retriever=base_retriever
    )
base_response = base_query_engine.query(query_str)
print("\n" + str(base_response))

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever
    )
response = query_engine.query(query_str)
print("\n" + str(response))

# print(base_response.get_formatted_sources(length=200))


# rerank_model = HuggingFaceEmbedding(model_name="BAAI/bge-reranker-base")  #error
rerank_model = "BAAI/bge-reranker-base"
rerank = SentenceTransformerRerank(
    top_n=6,
    model=rerank_model,
    )

auto_merging_engine = RetrieverQueryEngine.from_args(
    retriever=retriever, 
    node_postprocessors=[rerank],
    )

vector_store.client.load_collection(collection_name=collection_name)
rerank_response = auto_merging_engine.query(query_str)
print("\n" + str(rerank_response))

# print(auto_merging_response.get_formatted_sources(length=200))



vector_store.client.release_collection(collection_name=collection_name)
vector_store.client.close()  # Need to do this (otherwise Milvus container will hault when closing)
# del docstore  # MongoDB may frnot need to be manually closed.






