# PDF 读取和切片
from PyPDF2 import PdfReader

def parse_pdf(pdf_path:str)->str:
    """读取PDF，返回全文文本"""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text
    return full_text

def split_text(full_text:str,chunk_size:int=500,chunk_overla:int=100):
    """文本切片"""
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += (chunk_size-chunk_overla)
    return chunks