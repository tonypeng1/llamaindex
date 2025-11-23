
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymilvus import connections, db, MilvusClient


def check_if_milvus_database_exists(
        uri: str, 
        database_name: str
        ) -> bool:
    """
    Check if a specified database exists in the Milvus server.

    Args:
        uri (str): The URI of the Milvus server.
        database_name (str): The name of the database to check.

    Returns:
        bool: True if the database exists, False otherwise.
    """
    try:
        connections.connect(uri=uri)
        db_names = db.list_database()
        return database_name in db_names
    
    except Exception as e:
        print(f"An error occurred when checking if milvus database exists: {e}")

    finally:
        # Always disconnect from the server, even if an error occurred.
        connections.disconnect("default")


def check_if_milvus_collection_exists(
        uri: str, 
        db_name: str, 
        collect_name: str
        ) -> bool:
    """
    Check if a collection exists in a Milvus database.

    Parameters:
    uri (str): The URI of the Milvus server.
    db_name (str): The name of the database to check.
    collect_name (str): The name of the collection to check.

    Returns:
    bool: True if the collection exists, False otherwise.
    """
    # Create a Milvus client
    client = MilvusClient(
        uri=uri, 
        db_name=db_name)

    try:
        # List all collections in the database
        collect_names = client.list_collections()

    except Exception as e:
        print(f"An error occurred when checking if milvus collection exists: {e}")

    finally:
        # Ensure the client is closed, even if an error occurs
        client.close()

    # Check if the collection name is in the list of collections
    return collect_name in collect_names


def create_database_milvus(
        uri: str, 
        db_name: str,
        ):
    """
    Create a new database in Milvus.

    This function will attempt to connect to Milvus at the provided URI, create a new 
    database with the provided name, and then disconnect. If any error occurs during 
    this process, an exception will be raised.

    Parameters:
    uri (str): The URI of the Milvus server.
    db_name (str): The name of the database to create.

    Raises:
    Exception: If any error occurs during the database creation process.
    """
    try:
        connections.connect(uri=uri)
        db.create_database(db_name)

    except Exception as e:
        raise Exception(f"Failed to create database: {str(e)}")
    
    finally:
        connections.disconnect("default")


def milvus_collection_item_count(
        uri: str, 
        database_name: str, 
        collection_name: str,
        ) -> int:
    """
    This function returns the number of items in a specified collection in a Milvus database.

    Parameters:
    uri (str): The URI of the Milvus server.
    database_name (str): The name of the database.
    collection_name (str): The name of the collection.

    Returns:
    int: The number of items in the collection.
    """
    # Create a Milvus client
    client = MilvusClient(uri=uri, db_name=database_name)

    try:
        # Load the specified collection
        client.load_collection(collection_name=collection_name)

        # Query the count of items in the collection
        element_count = client.query(
            collection_name=collection_name,
            output_fields=["count(*)"],
        )

        # Return the count of items
        return element_count[0]['count(*)']

    except Exception as e:
        print(f"An error occurred when counting the items in milvus collection: {e}")

    finally:
        # Close the connection to the Milvus server
        client.close()


def check_if_milvus_database_collection_exist(
        uri: str, 
        database_name: str, 
        collection_name: str
        ) -> bool:
    """
    Check if a given database and collection (with populated items) exist in the Milvus database.
    If not, return Ture (it will need to be saved).

    1. If the database does not exist, create one.
    2. If the collection does not exist, return True (it will need to be saved).
    3. If the collection exists but has no items, also return True (it will need to be saved).

    Parameters:
    uri (str): The URI of the Milvus database.
    database_name (str): The name of the database to check for.
    collection_name (str): The name of the collection to check for.

    Returns:
    bool: True if the collection needs to be saved, False otherwise.
    """
    if not check_if_milvus_database_exists(uri, database_name):
        create_database_milvus(uri, database_name)

    if not check_if_milvus_collection_exists(uri, database_name, collection_name):
        return True  # Collection needs to be saved

    # Collection exists, check if it has any items
    num_count = milvus_collection_item_count(uri, database_name, collection_name)

    return num_count == 0  # return True if 0 items in the collection (collection needs to be saved)


def check_if_mongo_database_exists(uri, _database_name) -> bool:
    client = MongoClient(uri)
    db_names = client.list_database_names()
    client.close()
    return _database_name in db_names


def check_if_mongo_namespace_exists(
        uri: str, 
        db_name: str, 
        namespace: str,
        ) -> bool:
    """
    Function to check if a namespace exists in a MongoDB database.

    Parameters:
    uri (str): The MongoDB connection URI.
    db_name (str): The name of the database to check.
    namespace (str): The namespace to check for.

    Returns:
    bool: True if the namespace exists in the database, False otherwise.

    Raises:
    PyMongoError: If there is an error connecting to the MongoDB server.
    """

    try:
        client = MongoClient(uri)
        db = client[db_name]
        collection_names = db.list_collection_names()

        # Check if the namespace exists in the database (Choose from the 3 in the list)
        return namespace + "/data" in collection_names  
    
    except PyMongoError as e:
        # Handle any errors that occur during the database operation
        print(f"Error connecting to MongoDB: {e}")

    finally:
        # Ensure that the database connection is closed even if an error occurs
        client.close()  


def check_if_mongo_database_namespace_exist(
        uri: str, 
        database_name: str, 
        collection_name: str
        ) -> bool:
    """
    Check if a database and collection namespace exist in MongoDB. If not,
    return True to indicate that the data needs to be saved.

    Args:
        uri (str): The connection URI for the MongoDB server.
        database_name (str): The name of the database to check.
        collection_name (str): The name of the collection to check.

    Returns:
        bool: True if the database and collection namespace do not exist, False otherwise.
    """
    # Check if the database exists
    if check_if_mongo_database_exists(uri, database_name):
        # If the database exists, check if the collection exists
        if check_if_mongo_namespace_exists(uri, database_name, collection_name):
            return False  # The database and collection namespace exist (no need to save)
    return True  # The database and collection namespace do not exist (need to save)
