from llama_index.core.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
    )

from pymongo import MongoClient
from pymilvus import connections, db, MilvusClient, DataType


DEFAULT_BATCH_SIZE = 100
MILVUS_ID_FIELD = "id"


def check_if_milvus_database_exists(uri, _database_name) -> bool:
    connections.connect(uri=uri)
    db_names = db.list_database()
    connections.disconnect("default")
    return _database_name in db_names

def check_if_milvus_collection_exists(uri, db_name, collect_name) -> bool:
    client = MilvusClient(
        uri=uri,
        db_name=db_name
        )
    # client.load_collection(collection_name=collect_name)
    collect_names = client.list_collections()
    client.close()
    return collect_name in collect_names


def create_database_milvus(uri, db_name):
    """
    Only create a new database (collection not created yet).
    """
    connections.connect(uri=uri)
    db.create_database(db_name)
    connections.disconnect("default")


def create_database_collection_milvus_no_dynamic(
        uri, 
        _db_name, 
        _collection_name, 
        _dimension):
    """
    Only create a new database (collection not created yet).
    """
    connections.connect(uri=uri)
    db.create_database(_db_name)

    client = MilvusClient(
        uri=uri,
        db_name=_db_name,
        )

    schema = MilvusClient.create_schema(
        auto_id=False, 
        enable_dynamic_field=False,
    )
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        max_length=65535,
        is_primary=True,
    )
    schema.add_field(
        field_name=DEFAULT_EMBEDDING_KEY,
        datatype=DataType.FLOAT_VECTOR,
        dim=_dimension,
    )

    client.create_collection(
        collection_name=_collection_name,
        dimension=_dimension,
        primary_field_name=MILVUS_ID_FIELD,
        vector_field_name=DEFAULT_EMBEDDING_KEY,
        id_type="string",
        metric_type="IP",
        max_length=65_535,
        consistency_level="Session",
        schema=schema
        )

    connections.disconnect("default")


def milvus_collection_item_count(uri, 
                                 _database_name, 
                                 _collection_name) -> int:
    client = MilvusClient(
        uri=uri,
        db_name=_database_name
        )
    client.load_collection(collection_name=_collection_name)
    element_count = client.query(
        collection_name=_collection_name,
        output_fields=["count(*)"],
        )
    client.close()
    return element_count[0]['count(*)']


def check_if_milvus_database_collection_exist(
        uri, 
        _database_name, 
        _collection_name,
        ):
                                                                           
    save_ind = True
    if check_if_milvus_database_exists(uri, _database_name):
        if check_if_milvus_collection_exists(uri, _database_name, _collection_name):
            num_count = milvus_collection_item_count(uri, _database_name, _collection_name)
            if num_count > 0:  # account for the case of 0 item in the collection
                save_ind = False
    else:
        create_database_milvus(uri, _database_name)

    return save_ind


def check_if_milvus_database_collection_exist_create_both(
        uri, 
        _database_name, 
        _collection_name,
        _dimension,
        ):
                                                                           
    save_ind = True
    if check_if_milvus_database_exists(uri, _database_name):
        if check_if_milvus_collection_exists(uri, _database_name, _collection_name):
            num_count = milvus_collection_item_count(uri, _database_name, _collection_name)
            if num_count > 0:  # account for the case of 0 item in the collection
                save_ind = False
    else:
        # create_database_milvus(uri, _database_name)
        create_database_collection_milvus_no_dynamic(uri, 
                                          _database_name, 
                                          _collection_name, 
                                          _dimension)
    return save_ind


def check_if_mongo_database_exists(uri, _database_name) -> bool:
    client = MongoClient(uri)
    db_names = client.list_database_names()
    client.close()
    return _database_name in db_names


def check_if_mongo_namespace_exists(uri, db_name, namespace) -> bool:

    client = MongoClient(uri)
    db = client[db_name]
    collection_names = db.list_collection_names()
    client.close()
    return namespace + "/data" in collection_names  # Choose from 3 in the list


def check_if_mongo_database_namespace_exist(
        uri, 
        _database_name, 
        _collection_name) -> bool:
    
    add_doc = True
    if check_if_mongo_database_exists(uri, _database_name):
        if check_if_mongo_namespace_exists(uri, _database_name, _collection_name):
            add_doc = False
    return add_doc