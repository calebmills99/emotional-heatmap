import os
from striprtf.striprtf import rtf_to_text
from PyPDF2 import PdfReader
import docx
import markdown
from bs4 import BeautifulSoup
import chardet

def read_file_content(file_path: str) -> str:
    """
    Reads content from various file formats and returns extracted text.
    Supported formats: .txt, .pdf, .rtf, .docx, .md, .svt (treated as xml/text)
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        if ext == '.txt' or ext == '.svt':
            return _read_text_file(file_path)
        elif ext == '.pdf':
            return _read_pdf_file(file_path)
        elif ext == '.rtf':
            return _read_rtf_file(file_path)
        elif ext == '.docx':
            return _read_docx_file(file_path)
        elif ext == '.md':
            return _read_md_file(file_path)
        else:
            # Try reading as plain text for unknown extensions
            try:
                return _read_text_file(file_path)
            except Exception:
                raise ValueError(f"Unsupported file format: {ext}")
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def _read_text_file(file_path: str) -> str:
    # Detect encoding first
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] or 'utf-8'
        
    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        return f.read()

def _read_pdf_file(file_path: str) -> str:
    text = ""
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def _read_rtf_file(file_path: str) -> str:
    with open(file_path, 'r', errors='ignore') as f:
        content = f.read()
        return rtf_to_text(content)

def _read_docx_file(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def _read_md_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, features="html.parser")
        return soup.get_text()
