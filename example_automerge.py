import os

from llama_index.core import (
                        Document,
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
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.readers.file import PyMuPDFReader

import openai


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
        print(f"Item number: {i+1}")
        print(f"Score: {score}")
        # print(f"File Name: {file_name}")
        # print(f"File Path: {file_path}")
        print(f"Id: {chunk_id}")
        print("\nExtracted Content:")
        print(text_content)
        # print("\n" + "=" * 40 + " End of Result " + "=" * 40 + "\n")
        print("\n\n")


# Set OpenAI API key, LLM, and embedding model
openai.api_key = os.environ['OPENAI_API_KEY']
llm = OpenAI(model="gpt-3.5-turbo")

# create the hierarchical node parser w/ default settings
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)


# Load and parse document
loader = PyMuPDFReader()
docs0 = loader.load(file_path=Path("./data/llama/llama_2.pdf"))
document = Document(text="\n\n".join([doc.text for doc in docs0]))

nodes = node_parser.get_nodes_from_documents([document])
leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)

nodes_by_id = {node.node_id: node for node in nodes}
parent_node = nodes_by_id[leaf_nodes[20].parent_node.node_id]


# Assign docstore, vector store, and storage context
docstore = SimpleDocumentStore()

# insert nodes into docstore
docstore.add_documents(nodes)

# define storage context (will include vector store by default too)
storage_context = StorageContext.from_defaults(
    docstore=docstore
    )

# Create and save index
base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
)


# Define retrievers
base_retriever = base_index.as_retriever(
    similarity_top_k=6,
    )
retriever = AutoMergingRetriever(
    base_retriever, 
    storage_context, 
    verbose=True
    )

query_str = (
    "What could be the potential outcomes of adjusting the amount of safety"
    " data used in the RLHF stage?"
)

base_nodes_retrieved = base_retriever.retrieve(query_str)
print_retreived_nodes(base_nodes_retrieved)

nodes_retrieved = retriever.retrieve(query_str)
print_retreived_nodes(nodes_retrieved)



base_query_engine = RetrieverQueryEngine.from_args(base_retriever)
base_response = base_query_engine.query(query_str)
print("\n" + str(base_response))

query_engine = RetrieverQueryEngine.from_args(retriever)
response = query_engine.query(query_str)
print("\n" + str(response))








