import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""

    doc = fitz.open(pdf_path)   # ✅ open PDF file, NOT text
    for page in doc:
        text += page.get_text()

    doc.close()
    return text
