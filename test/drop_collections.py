from pymilvus import connections, utility, MilvusClient
from pymongo import MongoClient

# Configuration
uri_milvus = "http://localhost:19530"
uri_mongo = "mongodb://localhost:27017/"
database_name = "Rag_anything_mineru"
collection_name_vector = "openai_embedding_3_small_parse_method_jason"
collection_name_summary = "openai_embedding_3_small_parse_method_jason_summary"

print(f"🗑️ Dropping Milvus collection: {collection_name_vector} in DB: {database_name}")
try:
    # Specify the db_name in the client
    client = MilvusClient(uri=uri_milvus, db_name=database_name)
    if collection_name_vector in client.list_collections():
        client.drop_collection(collection_name=collection_name_vector)
        print(f"✅ Milvus collection '{collection_name_vector}' dropped.")
    else:
        print(f"ℹ️ Milvus collection '{collection_name_vector}' did not exist in DB '{database_name}'.")
    client.close()
except Exception as e:
    print(f"❌ Error dropping Milvus collection: {e}")

print(f"🗑️ Dropping MongoDB database: {database_name}")
try:
    mongo_client = MongoClient(uri_mongo)
    if database_name in mongo_client.list_database_names():
        mongo_client.drop_database(database_name)
        print(f"✅ MongoDB database '{database_name}' dropped.")
    else:
        print(f"ℹ️ MongoDB database '{database_name}' did not exist.")
    mongo_client.close()
except Exception as e:
    print(f"❌ Error dropping MongoDB database: {e}")
