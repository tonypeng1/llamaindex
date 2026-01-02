from config import get_article_info, DATABASE_CONFIG, EMBEDDING_CONFIG, ACTIVE_ARTICLE
from utils import get_database_and_llamaparse_collection_name, get_llamaparse_vector_store_docstore_and_storage_context

article_info = get_article_info()
article_directory = article_info["directory"]
article_name = article_info["filename"]
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

# Also connect to summary docstore if needed
from utils import get_summary_storage_context
storage_context_summary = get_summary_storage_context(uri_mongo, database_name, collection_name_summary)
summary_docstore = storage_context_summary.docstore

search_ids = [
    "c0e065ea-e421-42d4-853e-194d2d6152c7",
    "9fd2cdad-c3b4-4226-bc36-868e6ba01051",
]

found = {sid: [] for sid in search_ids}

print(f"Vector docstore has {len(getattr(vector_docstore, 'docs', {}))} entries")
print(f"Summary docstore has {len(getattr(summary_docstore, 'docs', {}))} entries")

# Helper to inspect an object for id matches

def inspect_obj(obj, sid):
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and sid in v:
                    return True
        else:
            # attributes
            for attr in dir(obj):
                if attr.startswith('__'):
                    continue
                try:
                    val = getattr(obj, attr)
                    if isinstance(val, str) and sid in val:
                        return True
                except Exception:
                    continue
    except Exception:
        return False
    return False

for k, v in getattr(vector_docstore, 'docs', {}).items():
    for sid in search_ids:
        if sid in str(k) or inspect_obj(v, sid):
            found[sid].append(('vector', k))

for k, v in getattr(summary_docstore, 'docs', {}).items():
    for sid in search_ids:
        if sid in str(k) or inspect_obj(v, sid):
            found[sid].append(('summary', k))

print("Search results:")
for sid, locations in found.items():
    print(f"  {sid}: {locations}")

# Print the text and metadata for found nodes
for sid, locations in found.items():
    for loc_type, key in locations:
        doc = None
        if loc_type == 'vector':
            doc = getattr(vector_docstore, 'docs', {}).get(key)
        else:
            doc = getattr(summary_docstore, 'docs', {}).get(key)
        print(f"\n--- {sid} ({loc_type} - {key}) ---")
        try:
            text = getattr(doc, 'text', None) or getattr(doc, 'node', {}).get('text') if doc else None
        except Exception:
            text = None
        try:
            meta = getattr(doc, 'metadata', None) or getattr(doc, 'node', {}).get('metadata') if doc else None
        except Exception:
            meta = None
        print("Text:")
        print(text[:500] if text else "<no text>")
        print("\nMetadata:")
        print(meta)

# Exit code
import sys
sys.exit(0)

# Exit code
import sys
sys.exit(0)
