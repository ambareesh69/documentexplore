# Document Explorer

A tool for exploring and analyzing document content using NLP techniques.

## Setup and Installation

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/DocumentExplore.git
   cd DocumentExplore
   ```

2. **Create a virtual environment**
   
   For macOS/Linux:
   ```
   python -m venv venv
   source venv/bin/activate
   ```
   
   For Windows:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```
   streamlit run app.py
   ```

## Features

- Upload and analyze PDF and DOCX documents
- Extract key insights and visualize document content
- Compare multiple documents

## Requirements

- Python 3.6+
- All dependencies are listed in requirements.txt

## Overview

DocumentExplore automatically analyzes documents and provides insights by:
- Extracting text from PDF and DOCX files
- Breaking down content into manageable chunks
- Creating vector representations (embeddings) of the text
- Automatically clustering similar content using machine learning
- Identifying key topics using unsupervised learning
- Visualizing the results in an interactive web interface

## Components

- **Text Extraction**: Extracts text from PDF and DOCX files
- **Text Chunking**: Divides text into manageable chunks
- **Embedding Generation**: Creates TF-IDF vector representations
- **Clustering**: Groups similar content using K-means with dynamic topic number detection
- **Topic Naming**: Automatically names clusters based on document content
- **Visualization**: Interactive web interface to explore document insights

