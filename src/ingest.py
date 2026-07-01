"""
===========================================================
Ingestion Pipeline

Responsibility:
- Load PDFs
- Split into chunks
- Create embeddings
- Store into ChromaDB
===========================================================
"""

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma


DB_PATH = "database"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def ingest():

    print("Loading PDF...")

    loader = PyPDFLoader("data/AWS.pdf")

    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    print("\nSplitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    print("\nCreating Vector Database...")

    Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_PATH
    )

    print("\n✅ Knowledge Base Created!")


if __name__ == "__main__":
    ingest()
