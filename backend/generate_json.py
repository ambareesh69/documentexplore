# backend/generate_json.py
import os
import json

def generate_docexplore_json(title, description, data_file, cluster_names_file, similarity_threshold=0.8, charsPerPixel=20, output_file=None):
    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "docexplore.json")
    
    # Load the clustered data and cluster names
    with open(data_file, "r", encoding="utf-8") as f:
        clustered_data = json.load(f)
    
    if not clustered_data:
        print("Warning: clustered_data is empty. Check previous pipeline steps (embedding_generation.py, clustering.py).")
    
    with open(cluster_names_file, "r", encoding="utf-8") as f:
        cluster_names = json.load(f)
    
    # Organize data into clusters
    clusters_dict = {}
    for item in clustered_data:
        cid = item["cluster"]
        if cid not in clusters_dict:
            clusters_dict[cid] = []
        clusters_dict[cid].append(item)
    
    # Create the clusters list for the JSON
    clusters = []
    for cid, items in clusters_dict.items():
        cluster_name = cluster_names.get(str(cid), f"Cluster {cid}")
        clusters.append({
            "id": cid,
            "name": cluster_name,
            "items": items  # Include the full items, which contain embeddings
        })
    
    # Create the configuration
    config = {
        "title": title,
        "description": description,
        "similarity": similarity_threshold,
        "charsPerPixel": charsPerPixel,
        "clusters": clusters  # Include the clustered data directly
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"docexplore.json generated at {output_file}")

if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "clustered_embeddings.json")
    cluster_names_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "cluster_names.json")
    generate_docexplore_json(
        title="Document Insights",
        description="Explore key topics and insights extracted from the document.",
        data_file=data_file,
        cluster_names_file=cluster_names_file,
        similarity_threshold=0.8,
        charsPerPixel=20
    )