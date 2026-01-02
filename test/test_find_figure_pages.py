from config import get_article_info, DATABASE_CONFIG, EMBEDDING_CONFIG, ACTIVE_ARTICLE
from utils import get_database_and_llamaparse_collection_name, get_llamaparse_vector_store_docstore_and_storage_context
from rag_factory import _find_pages_for_reference

article_info = get_article_info()
article_directory = article_info["directory"]
embed_model_name = EMBEDDING_CONFIG["short_name"]
chunk_method = "mineru"
chunk_size = article_info["rag_settings"]["chunk_size"]
chunk_overlap = article_info["rag_settings"]["chunk_overlap"]
metadata = article_info["rag_settings"].get("metadata")

database_name, collection_name_vector, collection_name_summary = get_database_and_llamaparse_collection_name(
    article_directory,
    ACTIVE_ARTICLE,
    chunk_method,
    embed_model_name,
    "mineru",
    chunk_size,
    chunk_overlap,
    metadata
)

uri_milvus = DATABASE_CONFIG["milvus_uri"]
uri_mongo = DATABASE_CONFIG["mongo_uri"]
embed_dim = EMBEDDING_CONFIG["dimension"]

vector_store, vector_docstore, storage_context_vector = get_llamaparse_vector_store_docstore_and_storage_context(
    uri_milvus,
    uri_mongo,
    database_name,
    collection_name_vector,
    embed_dim
)

print("Finding pages for figure 4.1")
pages_41 = _find_pages_for_reference('figure', '4.1', vector_docstore)
print("pages for 4.1:", pages_41)
print("Finding pages for figure 4.3")
pages_43 = _find_pages_for_reference('figure', '4.3', vector_docstore)
print("pages for 4.3:", pages_43)

# Also check for integer '4' (in case matching behavior)
pages_4 = _find_pages_for_reference('figure', '4', vector_docstore)
print("pages for 4:", pages_4)
