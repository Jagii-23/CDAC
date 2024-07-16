from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, MilvusException
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')


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
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),  # Assuming 384-dim vectors for SentenceTransformer
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512)  # Add text field to store sentences
]
schema = CollectionSchema(fields, description="Text similarity collection")

# Create collection
collection_name = "text_similarity_collection"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
collection = Collection(name=collection_name, schema=schema)
print(f"Collection '{collection_name}' created.")

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Diverse sentences
sentences = [
    # Sports
    "Soccer is the most popular sport in the world.",
    "I enjoy watching basketball games on weekends.",
    "Tennis is a great way to stay fit and healthy.",
    "The Olympics bring athletes from all over the world.",
    "Cricket is a sport with a rich history and tradition.",
    "Swimming is an excellent full-body workout.",
    "Running in the park helps me clear my mind.",
    "The World Cup is the biggest event in football.",
    "Baseball games are fun to attend with friends.",
    "Golf requires patience and precision.",
    
    # Food
    "I love trying new recipes in the kitchen.",
    "Pizza is a favorite food for many people.",
    "Eating healthy is important for a balanced diet.",
    "Sushi is a traditional Japanese dish.",
    "Desserts like chocolate cake are my weakness.",
    "Cooking can be a relaxing hobby.",
    "Spicy food can be a delightful experience.",
    "A balanced meal includes proteins, vegetables, and carbs.",
    "I enjoy making homemade pasta from scratch.",
    "Food festivals are great for discovering new dishes.",
    
    # Politics
    "Voting is a crucial part of democracy.",
    "Political debates often reflect the country's issues.",
    "Elections determine the leadership of a nation.",
    "Public policies affect everyone in society.",
    "Political activism can lead to meaningful change.",
    "Understanding different viewpoints is important in politics.",
    "International relations shape global politics.",
    "Lawmakers create legislation for the public good.",
    "Campaigns aim to persuade voters to choose a candidate.",
    "Civic engagement includes activities like volunteering and voting.",
    
    # Friends
    "Spending time with friends is essential for happiness.",
    "Friendships provide support and companionship.",
    "Traveling with friends creates lasting memories.",
    "Good friends are there for you through thick and thin.",
    "Friendship requires mutual trust and respect.",
    "Sharing experiences with friends strengthens bonds.",
    "Friendship can help you overcome difficult times.",
    "Having a diverse group of friends enriches your life.",
    "Friends are like family you choose for yourself.",
    "Friendship is built on communication and understanding.",
    
    # Studies
    "Studying hard is the key to academic success.",
    "Research projects can deepen your understanding of a subject.",
    "Group study sessions can be more effective than studying alone.",
    "Learning new languages opens up many opportunities.",
    "Time management skills are essential for students.",
    "Critical thinking helps you analyze information effectively.",
    "Academic achievements can lead to career advancements.",
    "Libraries are great resources for students.",
    "Online courses offer flexible learning options.",
    "Studying abroad can be a life-changing experience."
]

# Compute embeddings
embeddings = model.encode(sentences)

# Insert data into Milvus
ids = [i for i in range(len(sentences))]
data = [ids, embeddings.tolist(), sentences]
collection.insert(data)
print(f"Inserted {len(sentences)} texts into collection '{collection_name}'.")

# Create an index for the collection
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 64},
    "metric_type": "L2"
}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()
print("Index created and collection loaded into memory.")

# Function to get synonyms using WordNet
def get_related_words(keyword):
    synonyms = set()
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

# Function to perform similarity search
def search_similar_texts(query_text, threshold=2.0):
    query_embedding = model.encode([query_text])[0]
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "vector", search_params, limit=10, output_fields=["id", "text"])

    # Filter results based on the distance threshold
    filtered_results = [hit for hit in results[0] if hit.distance <= threshold]
    return filtered_results

# Loop to handle multiple test cases
while True:
    # Get user input for the keyword
    keyword = input("Enter the keyword to filter results (e.g., 'sport'), or type 'exit' to quit: ").strip().lower()
    if keyword == 'exit':
        break

    # Get related words
    related_words = get_related_words(keyword)
    related_words.add(keyword)  # Include the original keyword

    # Perform a similarity search for the user's keyword and related words
    all_results = []
    for word in related_words:
        query_text = word
        results = search_similar_texts(query_text)
        all_results.extend(results)

    # Remove duplicates and ensure unique results
    unique_results = {}
    for hit in all_results:
        if hit.id not in unique_results or hit.distance < unique_results[hit.id].distance:
            unique_results[hit.id] = hit

    # Sort results by distance
    sorted_results = sorted(unique_results.values(), key=lambda x: x.distance)

    # Print only the top 3 results with ranking
    print("Search results after filtering:")
    for idx, hit in enumerate(sorted_results[:3]):
        rank = ["first", "second", "third"][idx]
        print(f"Rank: {rank}, ID: {hit.id}, Distance: {hit.distance:.4f}, Sentence: {hit.entity.get('text')}")

# Disconnect from Milvus
connections.disconnect("default")
print("Disconnected from Milvus")
