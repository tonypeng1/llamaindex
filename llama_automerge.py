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
                        )
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.milvus import MilvusVectorStore
import openai

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# Create vector database
collection_name = "merging_index"
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name=collection_name,
    dim=384,  # dim of HuggingFace "BAAI/bge-small-en-v1.5" embedding model
    overwrite=True  ## If True, erase all content in the collection
)

# load documents
documents = SimpleDirectoryReader("./data/andrew/").load_data()

# print(type(documents), "\n")
# print(len(documents), "\n")
# print(type(documents[0]))
# print(documents[0])

document = Document(text="\n\n".join([doc.text for doc in documents]))

# create the hierarchical node parser w/ default settings
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)
nodes = node_parser.get_nodes_from_documents([document])
leaf_nodes = get_leaf_nodes(nodes)

print(leaf_nodes[30].text)
# print("\n\n".join([f"Node number: {i}, \n{leaf_nodes[i].text}" for i in range(15, 25)]))
# print("\n\n".join([f"Node number: {i}, \n{nodes[i].node_id}" for i in range(len(nodes))]))

nodes_by_id = {node.node_id: node for node in nodes}
parent_node = nodes_by_id[leaf_nodes[20].parent_node.node_id]

# print(parent_node.text)

# Set up LLM and embedding model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.node_parser = node_parser

# Use Milvus as the storage context and add nodes to docstore (not to Milvus database)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
    )
storage_context.docstore.add_documents(nodes)

# for i in list(storage_context.docstore.get_all_ref_doc_info().keys()):
#     print(i)
# print(storage_context.docstore.get_node(leaf_nodes[0].node_id))

# Create new index 
automerging_index = VectorStoreIndex(
    nodes=leaf_nodes, 
    storage_context=storage_context, 
    )

# Save to Milvus database
vector_store.client.load_collection(collection_name=collection_name)
element_count = vector_store.client.query(
    collection_name=collection_name,
    output_fields=["count(*)"],
    )


# Check if index is already saved in Milvus database
# vector_store.client.load_collection(collection_name=collection_name)
# element_count = vector_store.client.query(
#     collection_name=collection_name,
#     output_fields=["count(*)"],
#     )
# if element_count[0]['count(*)'] == 0:
#     # Create new index and save to Milvus database
#     automerging_index = VectorStoreIndex(
#         nodes=leaf_nodes, 
#         storage_context=storage_context, 
#         # service_context=auto_merging_context,
#         )
# else:  # load from Milvus database
#     automerging_index = VectorStoreIndex.from_vector_store(
#         vector_store=vector_store
#     )

# Create the retriever
base_retriever = automerging_index.as_retriever(
    similarity_top_k=12
)
retriever = AutoMergingRetriever(
    vector_retriever=base_retriever, 
    storage_context=automerging_index.storage_context, 
    verbose=True
)

query_str = "What is the importance of networking in AI?"

retrieved_base = base_retriever.retrieve(query_str)
retrieved = retriever.retrieve(query_str)

# Loop through each NodeWithScore in the response
for node_with_score in retrieved_base:
    node = node_with_score.node  # The TextNode object
    score = node_with_score.score  # The similarity score
    chunk_id = node.id_  # The chunk ID

    # Extract the relevant metadata from the node
    file_name = node.metadata.get("file_name", "Unknown")
    file_path = node.metadata.get("file_path", "Unknown")

    # Extract the text content from the node
    text_content = node.text if node.text else "No content available"

    # Print the results in a user-friendly format
    print(f"Score: {score}")
    print(f"File Name: {file_name}")
    print(f"Id: {chunk_id}")
    print("\nExtracted Content:")
    print(text_content)
    print("\n" + "=" * 40 + " End of Result " + "=" * 40 + "\n")


# rerank_model = HuggingFaceEmbedding(model_name="BAAI/bge-reranker-base")  #error
rerank = SentenceTransformerRerank(
    top_n=6,
    model="BAAI/bge-reranker-base",
    )

auto_merging_engine = RetrieverQueryEngine.from_args(
    retriever=retriever, 
    node_postprocessors=[rerank],
    )

vector_store.client.load_collection(collection_name=collection_name)
auto_merging_response = auto_merging_engine.query(
    "What is the importance of networking in AI?"
)


vector_store.client.release_collection(collection_name=collection_name)
vector_store.client.close()  # Need to do this (otherwise Milvus container will hault when closing)







