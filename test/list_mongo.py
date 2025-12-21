from pymongo import MongoClient

uri_mongo = "mongodb://localhost:27017/"
client = MongoClient(uri_mongo)

print("Listing MongoDB Databases:")
for db_name in client.list_database_names():
    print(f"Database: {db_name}")
    db = client[db_name]
    for coll_name in db.list_collection_names():
        print(f"  - Collection: {coll_name}")
