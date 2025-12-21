from pymilvus import connections, MilvusClient
from pymongo import MongoClient

# Configuration
uri_milvus = "http://localhost:19530"
uri_mongo = "mongodb://localhost:27017/"
database_name = "Rag_anything_mineru"
collection_name_vector = "openai_embedding_3_small_parse_method_jason"

print(f"🔍 Checking metadata in MongoDB DB: {database_name}, Collection: {collection_name_vector}")
try:
    mongo_client = MongoClient(uri_mongo)
    db = mongo_client[database_name]
    collection = db[collection_name_vector]
    
    # Get first 5 documents
    docs = list(collection.find().limit(5))
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        # The TextNode is stored in the 'metadata' or 'text' field depending on how LlamaIndex stores it
        # Actually, LlamaIndex stores the whole node as a JSON in the docstore
        # Let's just print the keys
        print(f"Keys: {doc.keys()}")
        if 'metadata' in doc:
            print(f"Metadata: {doc['metadata']}")
        # In LlamaIndex MongoDB docstore, the node is often in a 'data' field or similar
        # Let's print the whole doc but truncated
        # print(f"Full doc: {str(doc)[:500]}...")
        
except Exception as e:
    print(f"❌ Error: {e}")
