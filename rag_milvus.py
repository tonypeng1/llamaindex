
import logging
import os
import sys
import textwrap

from llama_index.core import (
                        Document,
                        SimpleDirectoryReader,
                        StorageContext,
                        VectorStoreIndex,
                        load_index_from_storage,
                        )
                        
from llama_index.vector_stores.milvus import MilvusVectorStore
import openai


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_key = os.environ['OPENAI_API_KEY']

# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

vector_store = MilvusVectorStore(
    uri="http://localhost:19530", 
    dim=1536, 
    # overwrite=True  ## If True, erase all content in the collection
)

# Use Milvus as the storage context
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
    )

# # Create new index and save to Milvus database ()
# index = VectorStoreIndex.from_documents(
#     documents, 
#     storage_context=storage_context
# )

# Load index from "llamaindex" collection to RAM
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store
    )

# index = load_index_from_storage(storage_context)  # not working

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")  ## This command still needs connection to Milvus 
print(response)

vector_store.client.load_collection(collection_name="llamacollection") 
vector_store.client.close()  # Need to close connection (otherwise Milvus server will hault)

print("Document ID:", documents[0].doc_id)


