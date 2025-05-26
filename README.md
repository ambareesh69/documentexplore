# DocumentExplore

A document analysis tool that extracts text from PDFs/DOCX files, generates embeddings, clusters content, and identifies key topics - all without requiring external API services.

## Overview

DocumentExplore automatically analyzes documents and provides insights by:
- Extracting text from PDF and DOCX files
- Breaking down content into manageable chunks
- Creating vector representations (embeddings) of the text
- Automatically clustering similar content using machine learning
- Identifying key topics using unsupervised learning
- Visualizing the results in an interactive web interface

## Setup

1. Create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place your PDF/DOCX files in the `reports/` directory.

4. Run the pipeline:
   ```
   python main.py
   ```

5. Open the web interface:
   ```
   streamlit run app.py
   ```

## Components

- **Text Extraction**: Extracts text from PDF and DOCX files
- **Text Chunking**: Divides text into manageable chunks
- **Embedding Generation**: Creates TF-IDF vector representations
- **Clustering**: Groups similar content using K-means with dynamic topic number detection
- **Topic Naming**: Automatically names clusters based on document content
- **Visualization**: Interactive web interface to explore document insights

