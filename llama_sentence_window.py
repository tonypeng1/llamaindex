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
from llama_index.core.indices.postprocessor import (
                        SentenceTransformerRerank,
                        MetadataReplacementPostProcessor,
                        )
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore

from pymilvus import connections, db, MilvusClient
# from pymongo import MongoClient
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

def print_retreived_nodes(_retriever):
    # Loop through each NodeWithScore in the retreived nodes
    for (i, node_with_score) in enumerate(_retriever):
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
        print("\nExtracted Content:\n")
        print(text_content)
        # print("\n" + "=" * 40 + " End of Result " + "=" * 40 + "\n")
        # print("\n")


def display_prompt_dict(_prompts_dict):
    for k, p in _prompts_dict.items():
        print(f"\nPrompt Key: {k} \nText:\n")
        print(p.get_template() + "\n")


def load_document_pdf(doc_link):
    loader = PyMuPDFReader()
    docs0 = loader.load(file_path=Path(doc_link))
    docs = Document(text="\n\n".join([doc.text for doc in docs0]))
    # print(type(documents), "\n")
    # print(len(documents), "\n")
    # print(type(documents[0]))
    # print(documents[0])
    return docs

def get_notes_from_document_sentence_window(docs, win_size):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=win_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
        )
    nodes = node_parser.get_nodes_from_documents([docs])
    return nodes


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


def build_sentence_window_index(
    art_link,
    save_ind,
    win_size,
    ):

    if save_ind == True:  # Only load and parse document if either index or docstore not saved.
        document = load_document_pdf(art_link)
        nodes = get_notes_from_document_sentence_window(document, win_size)

        # Create and save index (embedding) to Milvus database
        ind = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            )
    else:
        # Load from Milvus database
        ind = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
            )

    return ind

    
# # Set up logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set OpenAI API key, LLM, and embedding model
openai.api_key = os.environ['OPENAI_API_KEY']
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.llm = llm

# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# embed_model_dim = 384  # for bge-small-en-v1.5
# embed_model = "huggingface_embedding_bge_small"

Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
embed_model_dim = 1536  # for text-embedding-3-small
embed_model = "openai_embedding_3_small"

uri_milvus = "http://localhost:19530"

# Create database and collection names
article_dictory = "paul_graham"
article_name = "paul_graham_essay.pdf"

# article_dictory = "andrew"
# article_name = "eBook-How-to-Build-a-Career-in-AI.pdf"

article_link = "./data/" + article_dictory + "/" + article_name
chuck_method = "sentence_window"
window_size = 3

database_name = article_dictory + "_" + chuck_method
collection_name = embed_model + "_window_size_" + str(window_size)
# collection_name = embed_model + "_size_" + str(parent_2) + "_" + str(parent_1) + "_" + str(leaf)


# Check if index and docstore have already been saved to Milvus and MongoDB.

# In Milvus database, if the specific collection exists , do not save the index.
# Otherwise, create one.
save_index = check_if_milvus_database_collection_exist(database_name, collection_name)

# Initiate vector store (a new empty collection will be created in Milvus server)
vector_store = MilvusVectorStore(
    uri=uri_milvus,
    db_name=database_name,
    collection_name=collection_name,
    dim=embed_model_dim,  # dim of HuggingFace "BAAI/bge-small-en-v1.5" embedding model
    )

storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    )

# for i in list(storage_context.docstore.get_all_ref_doc_info().keys()):
#     print(i)
# print(storage_context.docstore.get_node(leaf_nodes[0].node_id))

# Get index
index = build_sentence_window_index(
    article_link,
    save_index,
    window_size,
)

# Get base retriever
similarity_top_k = 12
base_retriever = index.as_retriever(
    similarity_top_k=similarity_top_k,
    )

postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
    )

# The code below does not work (cannot put node_postprocessors here)
# window_retriever = index.as_retriever(
#     similarity_top_k=similarity_top_k,
#     node_postprocessors=[postproc],
#     )  

# query_str = "What are the keys to building a career in AI?"
# query_str = "What happened in New York?"
# query_str = "Describe everything that is mentioned about Interleaf one by one?"
# query_str = "Describe everything that is mentioned about Viaweb one by one?"
query_str = "Describe everything that is mentioned about Viaweb."
# query_str = "What happened at Interleaf?"
# query_str = "What happened at Interleaf and Viaweb?"
# query_str = "What is the importance of networking in AI?"
# query_str = (
#     "What could be the potential outcomes of adjusting the amount of safety"
#     " data used in the RLHF stage?"
# )

# Print retrieved nodes
vector_store.client.load_collection(collection_name=collection_name)

base_nodes = base_retriever.retrieve(query_str)
print_retreived_nodes(base_nodes)

window_nodes = postproc.postprocess_nodes(base_nodes)
print_retreived_nodes(window_nodes)


# Define reranker
rerank_model = "BAAI/bge-reranker-base"
rerank_top_n =6
rerank = SentenceTransformerRerank(
    top_n=rerank_top_n,
    model=rerank_model,
    )

rerank_nodes = rerank.postprocess_nodes(
    nodes=window_nodes,
    query_str=query_str,
    )
print_retreived_nodes(rerank_nodes)

# Since postproc and rerank are both post processors, there are no corresponding 
# retreivers, therefore, query enginers are not created using RetrieverQueryEngine.from_args()

# Directly use postprocessing in creating the query engines
window_engine = index.as_query_engine(
                            similarity_top_k=similarity_top_k,
                            node_postprocessors=[postproc],
                            # response_mode="compact",
                            )

compact_window_engine = index.as_query_engine(
                            similarity_top_k=similarity_top_k,
                            node_postprocessors=[postproc],
                            response_mode="compact",
                            )

tree_window_engine = index.as_query_engine(
                            similarity_top_k=similarity_top_k,
                            node_postprocessors=[postproc],
                            response_mode="tree_summarize",
                            )

accumulate_window_engine = index.as_query_engine(
                            similarity_top_k=similarity_top_k,
                            node_postprocessors=[postproc],
                            response_mode="accumulate",
                            )


rerank_engine = index.as_query_engine(
                            similarity_top_k=similarity_top_k,
                            node_postprocessors=[postproc, rerank],
                            )

compact_rerank_engine = index.as_query_engine(
                            similarity_top_k=similarity_top_k,
                            node_postprocessors=[postproc, rerank],
                            response_mode="compact",
                            )

tree_rerank_engine = index.as_query_engine(
                            similarity_top_k=similarity_top_k,
                            node_postprocessors=[postproc, rerank],
                            response_mode="tree_summarize",
                            )

accumulate_rerank_engine = index.as_query_engine(
                            similarity_top_k=similarity_top_k,
                            node_postprocessors=[postproc, rerank],
                            response_mode="accumulate",
                            )

# window_prompts_dict = window_engine.get_prompts()
# display_prompt_dict(window_prompts_dict)

# rerank_prompts_dict = rerank_engine.get_prompts()
# display_prompt_dict(rerank_prompts_dict)

# Prints window response
response = window_engine.query(query_str)
print("\nSENTENCE-WINDOW:\n\n" + str(response))

compact_response = compact_window_engine.query(query_str)
print("\nCOMPACT-SENTENCE-WINDOW:\n\n" + str(compact_response))

tree_response = tree_window_engine.query(query_str)
print("\nTREE-SENTENCE-WINDOW:\n\n" + str(tree_response))

accumulate_response = accumulate_window_engine.query(query_str)
print("\nACCUMULATE-SENTENCE-WINDOW:\n\n" + str(accumulate_response))

# Prints rerank response 
rerank_response = rerank_engine.query(query_str)
print("\nRE-RANK:\n\n" + str(rerank_response))

compact_rerank_response = compact_rerank_engine.query(query_str)
print("\nCOMPACT-RE-RANK:\n\n" + str(compact_rerank_response))

tree_rerank_response = tree_rerank_engine.query(query_str)
print("\nTREE-RE-RANK:\n\n" + str(tree_rerank_response))

accumulate_rerank_response = accumulate_rerank_engine.query(query_str)
print("\nACCUMULATE-RE-RANK:\n\n" + str(accumulate_rerank_response))

# print(rerank_response.get_formatted_sources(length=2000))



vector_store.client.release_collection(collection_name=collection_name)
vector_store.client.close()  # Need to do this (otherwise Milvus container will hault when closing)
# del docstore  # MongoDB may frnot need to be manually closed.






