from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530"
)

client.load_collection(
    collection_name="quick_setup",
    replica_number=1
)


client.list_partitions(collection_name="quick_setup")

# Get number of entities in a partition
client.get_partition_stats(
    collection_name="quick_setup",
    partition_name="partitionA"
    )

# Filter expression for IDs 11 through 15
filter_expression = "id in [11, 12, 13, 14, 15]"  

results = client.query(
    collection_name="quick_setup",
    filter=filter_expression,
    output_fields=["id", "vector", "color"]
)

for result in results:
    print(f"ID: {result['id']}, Vector: {result['vector']}, Color: {result['color']}")

# Query the number of entities in collectin "quick_setup"
client.query(
    collection_name="quick_setup", 
    # partition_name="partitionA",  # Do not support partition
    output_fields=["count(*)"]
    )

results = client.query(
    collection_name="quick_setup",
    filter="1 < id < 10",
    output_fields=["color"],
    limit=3
)

for result in results:
    print(f"\n\nID: {result['id']}, \
          \nColor: {result['color']}")

results = client.query(
    collection_name="quick_setup",
    filter='color == "pink_9298"',
    output_fields=["color"]
)

results = client.query(
    collection_name="quick_setup",
    filter='color like "pink%"',
    output_fields=["color"],
    limit=3
)

results = client.query(
    collection_name="quick_setup",
    # partition_name="partitionA",  # Not sure if suppoer partion, since results are the same
    filter='(color like "pink%") and (50 < id < 100)',
    output_fields=["color"],
    limit=3
)

client.query(
    collection_name="quick_setup",
    filter='(color like "red%") and (1 < id < 100)',
    output_fields=["count(*)"],
)


query_vectors = [
    [0.2580376395471989, -0.5023495712049978, 0.14414012509913835, -0.22286205330961354, 0.8029438446296592]
    ]

res = client.search(
    collection_name="quick_setup",
    data=query_vectors,
    output_fields=["vector", "color"],
    limit=3,
)

for result in res[0]:
    print(
        f"\n\nID: {result['id']}, \nDistance: {result['distance']}, \
        \nColor: {result['entity']['color']}, \
        \nVector: {result['entity']['vector']}"
        )
    
# Search with a filter expression using schema-defined fields
res = client.search(
    collection_name="quick_setup",
    data=query_vectors,
    filter='$meta["color"] like "red%"',
    output_fields=["vector", "color"],  # need to have this line
    limit=3
)

