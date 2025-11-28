
from pymongo import MongoClient
import json

def get_inclusive_schema_values():
    uri_mongo = "mongodb://localhost:27017/"
    # Based on check_node_in_mongo.py
    db_name = "paul_graham_paul_graham_essay_sentence_splitter"
    # The collection name in LlamaIndex MongoDocumentStore is usually <namespace>/data
    collection_namespace = "openai_embedding_3_small_chunk_size_256_chunk_overlap_64"
    collection_name = f"{collection_namespace}/data"

    try:
        client = MongoClient(uri_mongo)
        
        # Check if DB exists
        if db_name not in client.list_database_names():
            print(f"Database '{db_name}' not found. Available databases: {client.list_database_names()}")
            return

        db = client[db_name]
        all_collections = db.list_collection_names()
        print(f"All collections in {db_name}: {all_collections}")
        
        target_collection = None
        for col_name in all_collections:
            if col_name.endswith("/data"):
                col = db[col_name]
                if col.count_documents({"__data__.metadata.concept_categories": {"$exists": True}}) > 0:
                    print(f"Found enriched metadata in collection: {col_name}")
                    target_collection = col_name
                    break
        
        if target_collection:
            collection_name = target_collection
        else:
            print("Could not find any collection with 'concept_categories' metadata. Using default but expect empty results.")

        collection = db[collection_name]
        print(f"Querying collection: {collection_name} in database: {db_name}")
        
        count = collection.count_documents({})
        print(f"Total documents in collection: {count}")
        
        if count > 0:
            sample_doc = collection.find_one()
            print(f"Sample document keys: {list(sample_doc.keys())}")
            if '__data__' in sample_doc:
                 print(f"Sample document __data__ keys: {list(sample_doc['__data__'].keys())}")
                 if 'metadata' in sample_doc['__data__']:
                     print(f"Sample document __data__.metadata keys: {list(sample_doc['__data__']['metadata'].keys())}")

        # Check if any document has the relevant fields
        count_with_metadata = collection.count_documents({"__data__.metadata.concept_categories": {"$exists": True}})
        print(f"Documents with 'concept_categories': {count_with_metadata}")

        # Fields to query
        fields = [
            "concept_categories",
            "concept_importance",
            "advice_types",
            "advice_domains",
            "experience_periods",
            "experience_sentiments",
            "entity_roles",
            "entity_significance",
            "time_decades",
            "time_specificity"
        ]

        schema_values = {}

        for field in fields:
            # Check if we need to access via __data__
            # If the structure is { "__data__": { "metadata": { ... } } }
            mongo_field = f"__data__.metadata.{field}"
            
            # Try querying
            try:
                values = collection.distinct(mongo_field)
                # If empty, maybe try without __data__ just in case some docs are different
                if not values:
                     values = collection.distinct(f"metadata.{field}")
                
                values = [v for v in values if v]
                schema_values[field] = sorted(values)
            except Exception as e:
                print(f"Error querying field {field}: {e}")
                schema_values[field] = []

        print("\nInclusive Schema Definitions:")
        print(json.dumps(schema_values, indent=4))
        
        return schema_values

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    get_inclusive_schema_values()
