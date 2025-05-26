import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from config import MAX_KEYWORDS

def load_embeddings(embeddings_file):
    with open(embeddings_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def determine_optimal_clusters(X, min_clusters=3, max_clusters=50):
    """
    Dynamically determine the optimal number of clusters using silhouette score.
    This helps adapt the number of topics to the actual content.
    """
    # Set a reasonable range based on document size
    doc_size = X.shape[0]
    min_clusters = min(min_clusters, max(2, doc_size // 20))  # At least 1 cluster per 20 chunks
    max_clusters = min(max_clusters, max(min_clusters + 2, doc_size // 5))  # No more than 1 per 5 chunks
    
    # Don't compute too many variations for large documents (performance)
    if max_clusters > 30:
        step = (max_clusters - min_clusters) // 10
        cluster_range = list(range(min_clusters, max_clusters, max(1, step)))
    else:
        cluster_range = list(range(min_clusters, max_clusters))
    
    # We need at least a few options to compare
    if len(cluster_range) < 2:
        # Default to min_clusters if we don't have enough range
        return min_clusters
    
    # Standardize the data for better clustering
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute silhouette scores for different cluster counts
    silhouette_scores = []
    for n_clusters in cluster_range:
        try:
            # Skip if we don't have enough samples
            if X.shape[0] <= n_clusters:
                continue
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate silhouette score (measure of cluster quality)
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append((n_clusters, silhouette_avg))
            print(f"For n_clusters = {n_clusters}, silhouette score = {silhouette_avg:.3f}")
        except Exception as e:
            print(f"Error computing silhouette for {n_clusters} clusters: {e}")
    
    if not silhouette_scores:
        return min_clusters
        
    # Find the best number of clusters (highest silhouette score)
    best_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    
    # Print the selected number of topics
    print(f"Optimal number of topics for this document: {best_n_clusters}")
    return best_n_clusters

def perform_clustering(embeddings_data, num_clusters=None):
    # Extract embeddings into a numpy array (only if they exist and are non-empty)
    vectors = [item["embedding"] for item in embeddings_data if "embedding" in item and item["embedding"]]
    
    if not vectors:
        raise ValueError("No embeddings found. Please check that the embedding generation step produced output.")
    
    X = np.array(vectors)
    
    # Ensure that X is 2D. If it's 1D, reshape it.
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Determine the optimal number of clusters if not specified
    if num_clusters is None or num_clusters <= 0:
        num_clusters = determine_optimal_clusters(X)
    
    # Check if there are enough samples to form the requested clusters
    if X.shape[0] < num_clusters:
        # If we don't have enough samples, reduce the number of clusters
        print(f"Warning: Not enough data points ({X.shape[0]}) for {num_clusters} clusters.")
        num_clusters = max(2, X.shape[0] // 3)  # Ensure at least 2 clusters if possible
        print(f"Reducing to {num_clusters} clusters")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans

def save_clustered_data(embeddings_data, clusters, output_file):
    for i, cluster in enumerate(clusters):
        embeddings_data[i]["cluster"] = int(cluster)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, indent=4)
    print(f"Clustered data saved to {output_file}")

if __name__ == "__main__":
    embeddings_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "embeddings.json")
    output_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "clustered_embeddings.json")
    
    embeddings_data = load_embeddings(embeddings_file)
    try:
        # Use dynamic clustering to determine optimal number of topics
        clusters, model = perform_clustering(embeddings_data)
    except ValueError as e:
        print(f"Warning: {e}. Skipping clustering step.")
        clusters = []
    save_clustered_data(embeddings_data, clusters, output_file)
