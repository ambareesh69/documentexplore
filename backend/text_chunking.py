import os

def chunk_text(text, max_chars=4000):
    """Split text into chunks of up to max_chars characters."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start+max_chars]
        chunks.append(chunk)
        start += max_chars
    return chunks

def chunk_all_texts(input_file, output_file, max_chars=4000):
    with open(input_file, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    # Assume documents are separated by a marker.
    docs = full_text.split("\n\n=== NEW DOCUMENT ===\n\n")
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc, max_chars)
        all_chunks.extend(chunks)
    
    # Save each chunk with a separator marker for downstream processing.
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.strip() + "\n---CHUNK END---\n")
    
    print(f"Chunking complete, {len(all_chunks)} chunks saved to {output_file}")

if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "all_texts.txt")
    output_file = os.path.join(os.path.dirname(__file__), "..", "outputs", "chunks.txt")
    chunk_all_texts(input_file, output_file)
