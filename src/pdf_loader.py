"""
===========================================================
PDF Loader

Responsibility:
- Load PDFs using LangChain
===========================================================
"""

from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    def __init__(self, pdf_path):

        self.loader = PyPDFLoader(pdf_path)

    def load(self):

        documents = self.loader.load()

        return documents
