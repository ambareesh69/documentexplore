import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]

if __name__ == "__main__":
    # Test with example vectors.
    v1 = [0.1, 0.2, 0.3]
    v2 = [0.1, 0.2, 0.25]
    sim = compute_similarity(v1, v2)
    print(f"Cosine similarity: {sim}")
