# main.py
import subprocess

def run_data_extraction():
    print("\n--- Running data_extraction.py ---")
    subprocess.run(["python", "backend/data_extraction.py"], check=True)

def run_text_chunking():
    print("\n--- Running text_chunking.py ---")
    subprocess.run(["python", "backend/text_chunking.py"], check=True)

def run_embedding_generation():
    print("\n--- Running embedding_generation.py ---")
    subprocess.run(["python", "backend/embedding_generation.py"], check=True)

def run_clustering():
    print("\n--- Running clustering.py ---")
    subprocess.run(["python", "backend/clustering.py"], check=True)

def run_topic_naming():
    print("\n--- Running topic_naming.py ---")
    subprocess.run(["python", "backend/topic_naming.py"], check=True)

def run_generate_json():
    print("\n--- Running generate_json.py ---")
    subprocess.run(["python", "backend/generate_json.py"], check=True)

if __name__ == "__main__":
    try:
        run_data_extraction()
        run_text_chunking()
        run_embedding_generation()
        run_clustering()
        run_topic_naming()
        run_generate_json()
        print("\nAll steps completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: One of the scripts failed with exit code {e.returncode}.")
