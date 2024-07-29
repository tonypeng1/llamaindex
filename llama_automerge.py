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

from trulens_eval import Tru

from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.readers.file import PyMuPDFReader

from utility import (
                change_accumulate_engine_prompt_to_in_detail,
                change_default_engine_prompt_to_in_detail,
                change_tree_engine_prompt_to_in_detail,
                display_prompt_dict,
                get_article_link, 
                get_database_and_automerge_collection_name,
                get_default_query_engine_from_retriever,
                get_tree_query_engine_from_retriever,
                get_accumulate_query_engine_from_retriever,
                print_retreived_nodes,
                )
from database_operation import (
                check_if_milvus_database_collection_exist,
                check_if_mongo_database_namespace_exist
                )


def load_document_pdf(doc_link):
    loader = PyMuPDFReader()
    docs0 = loader.load(file_path=Path(doc_link))
    docs = Document(text="\n\n".join([doc.text for doc in docs0]))
    # print(type(documents), "\n")
    # print(len(documents), "\n")
    # print(type(documents[0]))
    # print(documents[0])
    return docs

def get_notes_from_document_automerge(docs, sizes):
    # create the hierarchical node parser w/ default settings
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=sizes
    )
    _nodes = node_parser.get_nodes_from_documents([docs])
    _leaf_nodes = get_leaf_nodes(_nodes)
    return _nodes, _leaf_nodes


def build_automerge_index_and_docstore(
    art_link,
    save_ind,
    add_doc,
    chunck_size,
    ):

    if save_ind or add_doc:  # Only load and parse document if either index or docstore not saved.
        _document = load_document_pdf(art_link)
        
        (_nodes, 
         _leaf_nodes) = get_notes_from_document_automerge(
                                                    _document, 
                                                    chunck_size
                                                    )
    if save_ind == True:
        # Create and save index (embedding) to Milvus database
        base_ind = VectorStoreIndex(
            nodes=_leaf_nodes,
            storage_context=storage_context,
            )
    else:
        # load from Milvus database
        base_ind = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
            )

    if add_doc == True:
        # Save document nodes to Mongodb docstore at the server
        storage_context.docstore.add_documents(_nodes)

    return base_ind


def get_automerge_retriever(
        _base_index,
        _similarity_top_k,
        _simple_ratio_thresh
        ):
    # Create the retriever
    _base_retriever = _base_index.as_retriever(
        similarity_top_k=_similarity_top_k
        )

    retrieve = AutoMergingRetriever(
        vector_retriever=_base_retriever, 
        # storage_context=automerging_index.storage_context,  # This does not work, results in dim mismatch. 
        storage_context=storage_context, 
        simple_ratio_thresh=_simple_ratio_thresh,
        verbose=True
        )
    
    return _base_retriever, retrieve



    
# # Set up logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set OpenAI API key, LLM, and embedding model
openai.api_key = os.environ['OPENAI_API_KEY']

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
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
article_dictory = "paul_graham"
article_name = "paul_graham_essay.pdf"

article_link = get_article_link(
                                article_dictory,
                                article_name
                                )

# article_dictory = "andrew"
# article_name = "eBook-How-to-Build-a-Career-in-AI.pdf"

# Create database and collection names
chunk_method = "automerge"
leaf = 256
parent_1 = 1024
parent_2 = 4096

automerge_chuck_sizes = [parent_2, parent_1, leaf]

# leaf = 128
# parent_1 = 512
# parent_2 = 2048

(database_name, 
collection_name) = get_database_and_automerge_collection_name(
                                                        article_dictory, 
                                                        chunk_method, 
                                                        embed_model_name,
                                                        automerge_chuck_sizes,
                                                        )

# Check if index has already been saved to Milvus database.
# In Milvus database, if the specific collection exists , do not save the index.
# Otherwise, create one.
uri_milvus = "http://localhost:19530"
save_index = check_if_milvus_database_collection_exist(uri_milvus, 
                                                       database_name, 
                                                       collection_name)

# Check if MongoDB already has the namespace
# In MongoDB, if the specific namespace exists, do not add document nodes to MongoDB.
uri_mongo = "mongodb://localhost:27017/"
add_document = check_if_mongo_database_namespace_exist(uri_mongo, 
                                                       database_name, 
                                                       collection_name)

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
# print(storage_context.docstore.get_node('0bc2db9b-9e4b-4426-9c13-7156ba0f3309').text)


# Get base index and build docstore
base_index = build_automerge_index_and_docstore(
                                        article_link,
                                        save_index,
                                        add_document,
                                        automerge_chuck_sizes,
                                        )

# Get retrievers and query engines
similarity_top_k = 12
rerank_model = "BAAI/bge-reranker-base"
simple_ratio_thresh = 0.4
rerank_top_n = 8

(base_retriever, 
 retriever) = get_automerge_retriever(
                                    base_index, 
                                    similarity_top_k,
                                    simple_ratio_thresh
                                    )

# query_str = "What are the keys to building a career in AI?"
query_str = "What are the thinkgs happened in New York?"
# query_str = "What happened in New York?"
# query_str = "What happened at Interleafe and Viaweb?"
# query_str = "What is the importance of networking in AI?"
# query_str = (
#     "What could be the potential outcomes of adjusting the amount of safety"
#     " data used in the RLHF stage?"
# )

vector_store.client.load_collection(collection_name=collection_name)

# Get retrieved nodes
base_nodes = base_retriever.retrieve(query_str)
automerge_nodes = retriever.retrieve(query_str)

# Define reranker
rerank_model = "BAAI/bge-reranker-base"
rerank_top_n = 6
rerank = SentenceTransformerRerank(
    top_n=rerank_top_n,
    model=rerank_model,
    )

# Get re-ranked fusion nodes
rerank_nodes = rerank.postprocess_nodes(
    nodes=automerge_nodes,
    query_str=query_str,
    )

# Print retrieved nodes
print_retreived_nodes("automerge base", base_nodes)
print_retreived_nodes("automerge", automerge_nodes)
print_retreived_nodes("rerank-automerge", rerank_nodes)


# Create default engines (query_mode="compact")
(base_engine, 
 engine) = get_default_query_engine_from_retriever(
                                        base_retriever,
                                        retriever,
                                        )

rerank_engine = RetrieverQueryEngine.from_args(
    retriever=retriever, 
    node_postprocessors=[rerank],
    )

# Create tree engines (query_mode="tree_summary")
(tree_base_engine,
 tree_engine) = get_tree_query_engine_from_retriever(
                                        base_engine,
                                        engine,
                                        )

tree_rerank_engine = RetrieverQueryEngine.from_args(
    retriever=retriever, 
    node_postprocessors=[rerank],
    response_mode="tree_summarize",
    )

# Create accumulate engines (query_mode="accumulate")
(accumulate_base_engine, 
 accumulate_engine) = get_accumulate_query_engine_from_retriever(
                                                base_retriever,
                                                retriever,
                                                )

accumulate_rerank_engine = RetrieverQueryEngine.from_args(
    retriever=retriever, 
    node_postprocessors=[rerank],
    response_mode="accumulate",
    )


# Change default engine promps to "in detail"
base_engine = change_default_engine_prompt_to_in_detail(base_engine) 
engine = change_default_engine_prompt_to_in_detail(engine) 
rerank_engine = change_default_engine_prompt_to_in_detail(rerank_engine) 

# Change tree engine prompt to "in detail"
tree_base_engine = change_tree_engine_prompt_to_in_detail(tree_base_engine) 
tree_engine = change_tree_engine_prompt_to_in_detail(tree_engine) 
tree_rerank_engine = change_tree_engine_prompt_to_in_detail(tree_rerank_engine) 

# Change accumulate engine prompt to "in detail"
accumulate_base_engine = change_accumulate_engine_prompt_to_in_detail(accumulate_base_engine) 
accumulate_engine = change_accumulate_engine_prompt_to_in_detail(accumulate_engine) 
accumulate_rerank_engine = change_accumulate_engine_prompt_to_in_detail(accumulate_rerank_engine) 

# window_prompts_dict = window_engine.get_prompts()
# type = "tree_fusion_engine:"
# display_prompt_dict(type.upper(), window_prompts_dict)

# Get responses 
base_response = base_engine.query(query_str)
response = engine.query(query_str)
rerank_response = rerank_engine.query(query_str)

tree_base_query_response = tree_base_engine.query(query_str)
tree_query_response = tree_engine.query(query_str)
tree_rerank_response = tree_rerank_engine.query(query_str)

accumulate_base_query_response = accumulate_base_engine.query(query_str)
accumulate_query_response = accumulate_engine.query(query_str)
accumulate_rerank_query_response = accumulate_rerank_engine.query(query_str)

# Print responses 
print("\nBASE-AUTOMERGE:\n\n" + str(base_response))
print("\nAUTOMERGE:\n\n" + str(response))
print("\nRERANK-AUTOMERGE:\n\n" + str(rerank_response))

print("\nTREE-BASE-AUTOMERGE:\n\n" + str(tree_base_query_response))
print("\nTREE-AUTOMERGE:\n\n" + str(tree_query_response))
print("\nTREE-RERANK-AUTOMERGE:\n\n" + str(tree_rerank_response))

print("\nACCUMULATE-BASE-AUTOMERGE:\n\n" + str(accumulate_base_query_response))
print("\nACCUMULATE-AUTOMERGE:\n\n" + str(accumulate_query_response))
print("\nACCUMULATE-RERANK-AUTOMERGE:\n\n" + str(accumulate_rerank_query_response))

# print(rerank_response.get_formatted_sources(length=2000))


vector_store.client.release_collection(collection_name=collection_name)
vector_store.client.close()  # Need to do this (otherwise Milvus container will hault when closing)
# del docstore  # MongoDB may frnot need to be manually closed.






