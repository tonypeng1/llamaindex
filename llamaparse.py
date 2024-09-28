from copy import deepcopy
import nest_asyncio
import os

from llama_index.core import (
                        Document,
                        Settings,
                        VectorStoreIndex,
                        )
from llama_index.core.node_parser import LlamaParseJsonNodeParser
from llama_index.core.schema import ImageDocument, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_parse import LlamaParse

from database_operation import (
                check_if_milvus_database_collection_exist,
                check_if_mongo_database_namespace_exist
                )
import openai

from utility import (
                get_article_link,
                get_database_and_llamaparse_collection_name,
                get_summary_storage_context,
                get_llamaparse_vector_store_docstore_and_storage_context, 
                )                


def load_document_llamaparse_jason(_json_objs):

    json_list = _json_objs[0]["pages"]

    documents = []
    for _, page in enumerate(json_list):
        documents.append(
            Document(
                text=page.get("text"),
                metadata=page,
            )
        )

    return documents


def get_nodes_from_document_llamaparse(
        _documnet, 
        _parse_method,
        ):
    
    if _parse_method == "jason":
        node_parser = LlamaParseJsonNodeParser(
                                    num_workers=16, 
                                    include_metadata=True
                                    )
        _nodes = node_parser.get_nodes_from_documents(documents=_documnet)  # this step may take a while
        _base_nodes, _objects = node_parser.get_nodes_and_objects(_nodes)

    return _base_nodes, _objects


def load_document_nodes_llamaparse(
    _json_objs,
    _parse_method,
    ):
    # Only load and parse document if either index or docstore not saved.
    _document = load_document_llamaparse_jason(
                                        _json_objs
                                        )
    (_base_nodes,
     _object) = get_nodes_from_document_llamaparse(
                                    _document, 
                                    _parse_method
                                    )
    return _base_nodes, _object


def load_image_text_nodes_llamaparse(_json_objs):

    anthropic_mm_llm = AnthropicMultiModal(max_tokens=1000,
                                           model="claude-3-5-sonnet-20240620",
                                           api_key=ANTHROPIC_API_KEY,
                                           )
    image_dicts = parser.get_images(_json_objs, 
                                    download_path="./images/"
                                    )
    img_text_nodes = []
    for image_dict in image_dicts:
        image_doc = ImageDocument(image_path=image_dict["path"])
        response = anthropic_mm_llm.complete(
            prompt="Describe the image in detail.",
            image_documents=[image_doc],
        )
        text_node = TextNode(text=str(response), 
                             metadata={"path": image_dict["path"]}
                             )
        img_text_nodes.append(text_node)

    return img_text_nodes


def create_and_save_llamaparse_vector_index_to_milvus_database(_base_nodes,
                                                               _objects,
                                                               _image_text_nodes):

    deepcopy_base_nodes = deepcopy(_base_nodes)
    deepcopy_objects = deepcopy(_objects)

    # Remove metadata from base nodes and objects before saving to vector store (othrewise max character 
    # in dynamic field in Milvus will be excceeded). Image text nodes have short metadata.
    for node in deepcopy_base_nodes:
        node.metadata = {}

    for obj in deepcopy_objects:
        obj.metadata = {}

    # In llamaoparse for Uber 2022, save 274 nodes to vector store (200 text nodes, 
    # 74 index nodes),and also save to MongoDB with 74 index nodes. The removal of
    # metadata does not seem to impact the details of the 74 index nodes in MongoDB.

    _index = VectorStoreIndex(
        nodes=deepcopy_base_nodes + deepcopy_objects + _image_text_nodes,
        storage_context=storage_context_vector,
        )
    
    return _index


nest_asyncio.apply()
LLAMA_CLOUD_API_KEY = os.environ['LLAMA_CLOUD_API_KEY']
MISTRAL_API_KEY = os.environ['MISTRAL_API_KEY']
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']



# Set OpenAI API key, LLM, and embedding model
openai.api_key = os.environ['OPENAI_API_KEY']
llm = OpenAI(model="gpt-4o", temperature=0.0)
Settings.llm = llm

# llm = MistralAI(
#     model="mistral-large-latest", 
#     temperature=0.0,
#     api_key=MISTRAL_API_KEY
#     )
# Settings.llm = llm

# llm = Anthropic(model="claude-3-5-sonnet-20240620", 
#                 api_key=ANTHROPIC_API_KEY,
#                 temperature=0.0)
# Settings.llm = llm

# Create embedding model
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
Settings.embed_model = embed_model
embed_model_dim = 1536  # for text-embedding-3-small
embed_model_name = "openai_embedding_3_small"

# Create article link
# article_dictory = "uber"
# article_name = "uber_10q_march_2022.pdf"

article_dictory = "attention"
article_name = "attention_all.pdf"

article_link = get_article_link(article_dictory,
                                article_name
                                )

# Create database and collection names
chunk_method = "llamaparse"
parse_method = "jason"

# parse_method = "markdown"

# Create database name and colleciton names
(database_name, 
collection_name_vector,
collection_name_summary) = get_database_and_llamaparse_collection_name(
                                                            article_dictory, 
                                                            chunk_method, 
                                                            embed_model_name, 
                                                            parse_method,
                                                            )

# Initiate Milvus and MongoDB database
uri_milvus = "http://localhost:19530"
uri_mongo = "mongodb://localhost:27017/"

# Check if the vector index has already been saved to Milvus database.
save_index_vector = check_if_milvus_database_collection_exist(uri_milvus, 
                                                       database_name, 
                                                       collection_name_vector
                                                       )

# Check if the vector document has already been saved to MongoDB database.
add_document_vector = check_if_mongo_database_namespace_exist(uri_mongo, 
                                                       database_name, 
                                                       collection_name_vector
                                                       )

# Check if the summary document has already been saved to MongoDB database.
add_document_summary = check_if_mongo_database_namespace_exist(uri_mongo, 
                                                       database_name, 
                                                       collection_name_summary
                                                       )

# Create vector store, vector docstore, and vector storage context
(vector_store,
 vector_docstore,
 storage_context_vector) = get_llamaparse_vector_store_docstore_and_storage_context(uri_milvus,
                                                                    uri_mongo,
                                                                    database_name,
                                                                    collection_name_vector,
                                                                    embed_model_dim
                                                                    )

# Create summary summary storage context
storage_context_summary = get_summary_storage_context(uri_mongo,
                                                    database_name,
                                                    collection_name_summary
                                                    )

# Load documnet nodes if either vector index or docstore not saved.
if save_index_vector or add_document_vector or add_document_summary: 

    parser = LlamaParse(
                api_key=LLAMA_CLOUD_API_KEY, 
                verbose=True
                )
    json_objs = parser.get_json_result(article_link)

    image_text_nodes = load_image_text_nodes_llamaparse(json_objs)

    (base_nodes,
     objects) = load_document_nodes_llamaparse(
                                            json_objs,
                                            parse_method,
                                            )

if save_index_vector == True:
    recursive_index = create_and_save_llamaparse_vector_index_to_milvus_database(
                                                                    base_nodes,
                                                                    objects, 
                                                                    image_text_nodes
                                                                    )

else:
    # Load from Milvus database
    recursive_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
        )

if add_document_vector == True:
    # Save document nodes to Mongodb docstore at the server
    storage_context_vector.docstore.add_documents(base_nodes + objects + image_text_nodes)

if add_document_summary == True:
    # Save document nodes to Mongodb docstore at the server
    storage_context_summary.docstore.add_documents(base_nodes + objects + image_text_nodes)

recursive_query_engine = recursive_index.as_query_engine(
                                            similarity_top_k=15, 
                                            verbose=True
                                            )

# query = "what is UBER Short-term insurance reserves reported in 2022?"
# query = "what is UBER's common stock subject to repurchase in 2021?"
# query = "What is UBER Long-term insurance reserves reported in 2021?"
# query = "What is the number of monthly active platform consumers in Q2 2021?"
# query = "What is the number of monthly active platform consumers in 2022?"
# query = "What is the number of trips in 2021?"
# query = "What is the free cash flow in 2021?"
# query = "What is the gross bookings of delivery in Q3 2021?"
# query = "What is the gross bookings in 2022?"
# query = "What is the value of mobility adjusted EBITDA in 2022?" 
# query = "What is the status of the classification of drivers?"
# query = "What is the comprehensive income (loss) attributable to Uber reported in 2021?"
# query = "What is the comprehensive income (loss) attributable to Uber Technologies reported in 2022?"
query = "What are the data shown in the bar graph titled 'Monthly Active Platform Consumers'?"
# query = "Can you tell me the page number on which the bar graph titled 'Monthly Active Platform Consumers' is located?"
# query = "What are the data shown in the bar graph titled 'Monthly Active Platform Consumers' on page 43?"
# query = "What is the Q2 2020 value shown in the bar graph titled 'Monthly Active Platform Consumers' on page 43?"
# query = "What are the main risk factors for Uber?"
# query = "What are the data shown in the bar graph titled 'Trips'?"
# query = "What are the data shown in the bar graph titled 'Gross Bookings'?"


response = recursive_query_engine.query(query)

print(response)

vector_store.client.release_collection(collection_name=collection_name_vector)
vector_store.client.close()  
