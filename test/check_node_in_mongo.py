
from pymongo import MongoClient

uri_mongo = "mongodb://localhost:27017/"
db_name = "paul_graham_paul_graham_essay_sentence_splitter"
collection_name = "openai_embedding_3_small_chunk_size_256_chunk_overlap_64"

print(f"Checking database: {db_name}")
print(f"Checking collection: {collection_name}")

try:
    client = MongoClient(uri_mongo)
    db = client[db_name]
    collections = db.list_collection_names()
    print(f"Collections in {db_name}: {collections}")
    
    target_nodes = [
        "206dcbdd-75db-4677-bfc1-4ad8cd2f8ee0",
        "6fe848df-c42b-4bcd-85df-eb6c4dfc20f5"
    ]
    
    for node_id in target_nodes:
        print(f"\nSearching for node {node_id} across ALL collections...")
        found = False
        for col_name in collections:
            # Skip metadata/ref_doc_info collections if you only care about data, 
            # but let's check everything just in case.
            col = db[col_name]
            doc = col.find_one({"id_": node_id})
            
            if doc:
                print(f"!!! FOUND in collection: {col_name}")
                print(f"Content preview: {doc.get('text', '')[:50]}...")
                found = True
                break # Stop searching for this node if found
        
        if not found:
            print(f"Node {node_id} NOT FOUND in any collection.")
            
            # REVERSE LOOKUP: Find who points to this missing node
            print(f"Performing REVERSE LOOKUP for {node_id} in {collection_name}/data...")
            main_col = db[collection_name + "/data"]
            
            # Search for the ID in the relationships field values
            # LlamaIndex relationships are stored as: relationships: { <type>: { node_id: ... } }
            # We'll try a regex search on the relationships field to be safe/broad
            
            query = {
                "$or": [
                    {"relationships.1.node_id": node_id}, # Next
                    {"relationships.2.node_id": node_id}, # Previous
                    {"relationships.3.node_id": node_id}, # Parent
                ]
            }
            
            referencing_doc = main_col.find_one(query)
            
            if referencing_doc:
                ref_id = referencing_doc.get("id_")
                print(f"!!! FOUND REFERENCE: Node {ref_id} points to missing node {node_id}")
                print(f"Referencing Node Content: {referencing_doc.get('text', '')[:50]}...")
                print(f"Relationships of referencing node: {referencing_doc.get('relationships')}")
            else:
                print(f"No node found that references {node_id} (checked Next/Prev/Parent relationships).")

except Exception as e:
    print(f"Error: {e}")
