import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from config import MAX_FEATURES

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Get the vector embedding for a text chunk using TF-IDF.
    Returns a dense vector representation of the text.
    """
    global _vectorizer, _fitted
    
    # Initialize vectorizer if not already done
    if '_vectorizer' not in globals():
        _vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
        _fitted = False
    
    # We need at least one document to fit the vectorizer
    if not _fitted:
        _vectorizer.fit([text])
        _fitted = True
    
    # Transform the text to get the embedding
    vector = _vectorizer.transform([text]).toarray()[0]
    
    # Convert to list for JSON serialization
    return vector.tolist()

def generate_embeddings(chunks_file, output_file):
    """
    Reads text chunks from a file, generates embeddings using TF-IDF,
    and writes the results to an output JSON file.
    
    Each chunk should be separated by the marker '---CHUNK END---'.
    """
    embeddings_data = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split the file content into individual chunks.
    chunks = [chunk.strip() for chunk in content.split("---CHUNK END---") if chunk.strip()]
    
    # Initialize and fit the vectorizer with all chunks
    global _vectorizer, _fitted
    _vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    _vectorizer.fit(chunks)
    _fitted = True
    
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)}...")
        embedding = get_embedding(chunk)
        if embedding is not None:
            embeddings_data.append({
                "id": idx,
                "text": chunk,
                "embedding": embedding
            })
        else:
            print(f"Skipping chunk {idx} due to embedding error.")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, indent=4)
    
    print(f"Generated embeddings for {len(embeddings_data)} chunks and saved to {output_file}")

if __name__ == "__main__":
    chunks_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "chunks.txt")
    output_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "embeddings.json")
    generate_embeddings(chunks_file, output_file)
