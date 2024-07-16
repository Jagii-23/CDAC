from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility, MilvusException
from sentence_transformers import SentenceTransformer
import numpy as np

# Connect to Milvus
try:
    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus")
except MilvusException as e:
    print(f"Failed to connect to Milvus: {e}")
    exit(1)

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)  # Assuming 384-dim vectors for SentenceTransformer
]
schema = CollectionSchema(fields, description="Text similarity collection")

# Create collection
collection_name = "text_similarity_collection"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema)

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example sentences
sentences = [
    "I love playing football",
    "I enjoy eating ice cream",
    "Soccer is my favorite sport"
]

# Compute embeddings
embeddings = model.encode(sentences)

# Insert data into Milvus
ids = [i for i in range(len(sentences))]
data = [ids, embeddings.tolist()]

collection.insert(data)

# Create an index for the collection
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 64},
    "metric_type": "L2"
}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()

# Function to perform similarity search
def search_similar_texts(query_text):
    query_embedding = model.encode([query_text])[0]
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "vector", search_params, limit=3, output_fields=["id"])
    return results

# Perform a similarity search for a query text
query_text = "I love soccer"
results = search_similar_texts(query_text)

print("Search results:")
for result in results:
    for hit in result:
        print(f"ID: {hit.id}, Distance: {hit.distance}, Sentence: {sentences[hit.id]}")

# Disconnect from Milvus
connections.disconnect("default")
