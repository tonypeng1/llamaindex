from pymilvus import connections, Collection, utility

uri_milvus = "http://localhost:19530"
db_name = "paul_graham_paul_graham_essay_sentence_splitter"
collection_name = "openai_embedding_3_small_chunk_size_256_chunk_overlap_64"
node_id = "206dcbdd-75db-4677-bfc1-4ad8cd2f8ee0"

print(f"Connecting to Milvus at {uri_milvus}...")
try:
    connections.connect(alias="default", uri=uri_milvus)
    print("Connected.")

    # Check if database exists (Milvus 2.x supports databases, but sometimes it's just 'default')
    # We'll try to use the db_name if possible, or check if the collection exists in 'default'
    
    # Note: pymilvus handling of databases varies by version. 
    # We'll try to list collections in the specific database if possible, or just use the connection.
    
    # In Milvus, you usually specify the db_name when connecting or using `using_database`
    # But let's try to just access the collection first.
    
    # Try to use the database
    try:
        from pymilvus import db
        db.using_database(db_name)
        print(f"Using database: {db_name}")
    except Exception as e:
        print(f"Could not switch to database {db_name}: {e}")
        print("Checking 'default' database instead...")
        db.using_database("default")

    if utility.has_collection(collection_name):
        print(f"Collection {collection_name} exists.")
        col = Collection(collection_name)
        col.load()
        
        print(f"Searching for node {node_id} in Milvus...")
        # Milvus usually stores the node_id in a field, often 'id' or 'doc_id' or 'node_id'
        # LlamaIndex default schema usually has 'id' as the primary key (VarChar)
        
        # Let's check the schema
        print("Schema:", col.schema)
        
        # Query for the ID
        # Assuming the primary key field is named 'id' or similar.
        pk_field = col.primary_field.name
        print(f"Primary key field: {pk_field}")
        
        res = col.query(expr=f'{pk_field} == "{node_id}"', output_fields=["*"])
        
        if res:
            print(f"!!! FOUND Node {node_id} in Milvus!")
            print(f"Data: {res[0]}")
        else:
            print(f"Node {node_id} NOT FOUND in Milvus collection {collection_name}.")
            
    else:
        print(f"Collection {collection_name} does NOT exist in the current database.")

except Exception as e:
    print(f"Error: {e}")
