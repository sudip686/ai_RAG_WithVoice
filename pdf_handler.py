import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader
import logging

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_WORKERS = 4

def process_image(image):
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        logging.warning(f"OCR failed on image: {e}")
        return ""

def extract_images_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            ocr_texts = list(executor.map(process_image, images))
        return [text for text in ocr_texts if text.strip()]
    except Exception as e:
        logging.warning(f"Error extracting images: {e}")
        return []

@lru_cache(maxsize=5)
def load_and_process_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    text_docs = loader.load()
    image_texts = extract_images_from_pdf(pdf_path)
    for img_text in image_texts:
        text_docs.append(type(text_docs[0])(page_content=img_text))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(text_docs)