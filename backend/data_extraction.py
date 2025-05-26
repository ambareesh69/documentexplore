import os
from pdfminer.high_level import extract_text
from docx import Document

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {pdf_path} - {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    try:
        doc = Document(docx_path)
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error extracting text from DOCX: {docx_path} - {e}")
        return ""

def extract_all_texts(input_dir):
    """Extract text from all PDF and DOCX files in the given directory."""
    texts = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
                texts.append(text)
            elif file.lower().endswith('.docx'):
                text = extract_text_from_docx(file_path)
                texts.append(text)
    return texts

if __name__ == "__main__":
    # Ensure your reports (PDF/DOCX) are stored in the 'reports' folder.
    input_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    texts = extract_all_texts(input_dir)
    
    # Create the outputs directory if it doesn't exist.
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all texts into one file, separated by a marker.
    output_file_path = os.path.join(output_dir, "all_texts.txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n\n=== NEW DOCUMENT ===\n\n".join(texts))
    print(f"Text extraction completed and saved to {output_file_path}")
