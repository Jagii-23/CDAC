from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusException
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch  # Import PyTorch for using torch.no_grad()

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
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)  # 768-dim vectors for BERT
]
schema = CollectionSchema(fields, description="Text similarity collection")

# Create collection
collection_name = "text_similarity_collection"
try:
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists, dropping it.")
        utility.drop_collection(collection_name)
    
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")
except MilvusException as e:
    print(f"Failed to create collection: {e}")
    exit(1)

# Initialize the tokenizer and model for text embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def text_to_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():  # Ensure torch is imported
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings[0]

# Insert data
texts = ["Hello world", "How are you?", "Goodbye", "I love programming", "Milvus is great for vector search"]
vectors = [text_to_vector(text) for text in texts]
ids = [i for i in range(len(texts))]

# Flatten the vectors list to a 1D list
flat_vectors = [vector for sublist in vectors for vector in sublist]

# Ensure ids and vectors are correctly formatted for Milvus
data = [ids, flat_vectors]

try:
    collection.insert(data)
    print(f"Inserted {len(texts)} texts into collection '{collection_name}'.")
except MilvusException as e:
    print(f"Failed to insert data: {e}")
    exit(1)

# Create index
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 64},
    "metric_type": "L2"
}

try:
    collection.create_index(field_name="vector", index_params=index_params)
    print("Index created.")
except MilvusException as e:
    print(f"Failed to create index: {e}")
    exit(1)

# Load collection to memory
try:
    collection.load()
    print("Collection loaded into memory.")
except MilvusException as e:
    print(f"Failed to load collection: {e}")
    exit(1)

# Perform a similarity search
query_text = "I enjoy coding"
query_vector = text_to_vector(query_text)

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}

try:
    results = collection.search(query_vector, "vector", search_params, limit=3)
    print("Search results:")
    for result in results:
        for hit in result:
            print(f"ID: {hit.id}, Distance: {hit.distance}")
except MilvusException as e:
    print(f"Failed to perform search: {e}")

# Retrieve and print some of the inserted data
try:
    results = collection.query(expr="id in [0, 1, 2, 3, 4]", output_fields=["id", "vector"])
    print("Retrieved data:")
    for result in results:
        print(f"ID: {result['id']}, Vector: {result['vector'][:5]}...")  # Print first 5 dimensions of the vector for brevity
except MilvusException as e:
    print(f"Failed to retrieve data: {e}")

# List all collections
try:
    collections = utility.list_collections()
    print("List of all collections:")
    print(collections)
except MilvusException as e:
    print(f"Failed to list collections: {e}")

# Delete collection
try:
    utility.drop_collection(collection_name)
    print(f"Collection '{collection_name}' deleted.")
except MilvusException as e:
    print(f"Failed to delete collection: {e}")
