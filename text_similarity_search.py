import spacy
import json
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, MilvusException

# Load the spaCy model for English language
spacy_model = spacy.load('en_core_web_lg')

# Connect to Milvus server
try:
    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus")
except MilvusException as e:
    print(f"Failed to connect to Milvus: {e}")
    exit(1)

# Define collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),  # Reduced to 128-dim vectors
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=256)  # For storing movie descriptions
]
schema = CollectionSchema(fields, description="Movies collection")

# Create collection
collection_name = "Movies"
try:
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists, dropping it.")
        utility.drop_collection(collection_name)
    
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")
except MilvusException as e:
    print(f"Failed to create collection: {e}")
    exit(1)

# Movie descriptions
descriptions = [
    "A pulse-pounding action film with explosive shootouts and high-octane chases",
    "An epic fantasy adventure set in a magical realm with mythical creatures and epic battles",
    "A gripping sci-fi thriller exploring the mysteries of outer space and extraterrestrial life",
    "A heartwarming family drama centered around love, loss, and the bonds that endure",
    "A mind-bending psychological thriller that keeps you on the edge of your seat",
    "A hilarious comedy filled with witty banter, slapstick humor, and outrageous antics",
    "A romantic escapade in the enchanting streets of Paris, weaving tales of love and destiny",
    "A spine-chilling horror story set in a haunted mansion with dark secrets lurking within",
    "A thought-provoking drama delving into the complexities of human relationships and morality",
    "A captivating mystery unfolding in a small town, where every resident has a hidden agenda"
]

description_vectors_list = []

for description in descriptions:
    doc = spacy_model(description)
    reduced_vector = doc.vector[:128].tolist()  # Reduce vector dimensions to 128
    entry = {"vector": reduced_vector, "description": description}
    description_vectors_list.append(entry)

# Convert to lists of ids, vectors, and descriptions
ids = list(range(len(descriptions)))
vectors = [entry['vector'] for entry in description_vectors_list]
desc = [entry['description'] for entry in description_vectors_list]

# Prepare data for insertion
data = [ids, vectors, desc]

# Insert data into the collection
try:
    collection.insert(data)
    print(f"Inserted {len(descriptions)} movie descriptions into collection '{collection_name}'.")
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

# Save data to JSON file
with open('dummy_data.json', 'w') as json_file:
    json.dump(description_vectors_list, json_file, indent=2)

print("JSON file created: dummy_data.json")
