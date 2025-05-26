import os
import json
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import re
from config import MAX_KEYWORDS

def name_cluster(text_samples, cluster_id):
    """
    Extract exact phrases from the document that best represent each topic cluster.
    No information is generated - only actual document content is used.
    """
    try:
        # Handle edge cases
        if len(text_samples) == 0:
            return f"Topic {cluster_id}"
 
        # Extract frequent phrases that actually appear in the text
        # Use n-grams to capture multi-word expressions
        vectorizer = CountVectorizer(
            stop_words='english',
            max_features=100,
            ngram_range=(1, 3),  # Allow up to 3-word phrases
            binary=False         # Count frequency
        )
        
        # Fit and transform on the actual text samples
        X = vectorizer.fit_transform(text_samples)
        
        # Get feature names (the actual phrases from the document)
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate frequency of each phrase across all samples
        phrase_counts = np.sum(X.toarray(), axis=0)
        
        # Sort phrases by frequency
        sorted_idx = np.argsort(phrase_counts)[::-1]
        top_phrases = [feature_names[i] for i in sorted_idx[:MAX_KEYWORDS*2]]

        # Use the most frequent multi-word phrases first (more informative)
        # Filter to include only multi-word phrases and single words if needed
        multi_word_phrases = [p for p in top_phrases if len(p.split()) > 1]
        single_words = [p for p in top_phrases if len(p.split()) == 1]
        
        # Prioritize multi-word phrases, but include single words if needed
        selected_phrases = []
        if multi_word_phrases:
            selected_phrases = multi_word_phrases[:min(MAX_KEYWORDS, len(multi_word_phrases))]
        
        # Add single words if we need more terms
        if len(selected_phrases) < MAX_KEYWORDS:
            needed = MAX_KEYWORDS - len(selected_phrases)
            selected_phrases.extend(single_words[:needed])
        
        # Simple, clean approach using only document content
        if not selected_phrases:
            return f"Topic {cluster_id}"
        
        # Format selected phrases, capitalize each one
        formatted_phrases = [phrase.capitalize() for phrase in selected_phrases[:MAX_KEYWORDS]]
        
        # Join the phrases with appropriate connector
        if len(formatted_phrases) > 1:
            # Use the most important phrase as the main topic
            # Connect remaining phrases with "&" which is a simple connector
            title = f"{formatted_phrases[0]} & {' '.join(formatted_phrases[1:])}"
        else:
            title = formatted_phrases[0]
        
        return title if title else f"Cluster {cluster_id}"
    except Exception as e:
        print(f"Error naming cluster {cluster_id}: {e}")
        return f"Cluster {cluster_id}"

def assign_cluster_names(clustered_file, output_file, samples_per_cluster=3):
    with open(clustered_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Group texts by cluster.
    clusters = {}
    for item in data:
        cluster = item.get("cluster")
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(item["text"])
    
    cluster_names = {}
    for cluster_id, texts in clusters.items():
        # Use all texts in the cluster for more accurate topic extraction
        sample_texts = texts
        cluster_name = name_cluster(sample_texts, cluster_id)
        cluster_names[str(cluster_id)] = cluster_name
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cluster_names, f, indent=4)
    
    print(f"Cluster names saved to {output_file}")

if __name__ == "__main__":
    clustered_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "clustered_embeddings.json")
    output_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "cluster_names.json")
    assign_cluster_names(clustered_file, output_file)
