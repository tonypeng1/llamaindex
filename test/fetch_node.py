
from pymongo import MongoClient
import json

uri_mongo = "mongodb://localhost:27017/"
client = MongoClient(uri_mongo)

db_name = "Rag_anything_mineru"
# The user mentioned "openai_embedding_3_small_parse_method_jason/data collection"
# In MongoDocumentStore, the collection is often just 'data' if the namespace is used as the DB or similar.
# Or the collection name itself might be 'openai_embedding_3_small_parse_method_jason.data'
# Let's list collections first to be sure.

db = client[db_name]
collections = db.list_collection_names()
print(f"Collections in {db_name}: {collections}")

target_id = "684133d5-c66c-4637-a205-12f132607a75"

found = False
# Sort collections to check 'data' before 'metadata' if possible, or just check all
for coll_name in sorted(collections, key=lambda x: 'data' not in x):
    coll = db[coll_name]
    doc = coll.find_one({"_id": target_id})
    if not doc:
        doc = coll.find_one({"id": target_id})
    
    if doc:
        print(f"\n🔍 Found document in collection: {coll_name}")
        # LlamaIndex MongoDocumentStore stores the serialized node in the '__data__' field
        if '__data__' in doc:
            node_data = doc['__data__']
            if isinstance(node_data, str):
                try:
                    node_data = json.loads(node_data)
                except Exception as e:
                    print(f"Error parsing __data__ string: {e}")
                    node_data = None
            
            if node_data and 'text' in node_data:
                print("--- TEXT START ---")
                print(node_data['text'])
                print("--- TEXT END ---")
                found = True
        elif 'json' in doc:
            try:
                node_data = json.loads(doc['json'])
                if 'text' in node_data:
                    print("--- TEXT START ---")
                    print(node_data['text'])
                    print("--- TEXT END ---")
                    found = True
            except:
                pass
        elif 'text' in doc:
            print("--- TEXT START ---")
            print(doc['text'])
            print("--- TEXT END ---")
            found = True
        
        if found:
            break

if not found:
    print(f"\n❌ Could not find document with _id or id: {target_id}")
