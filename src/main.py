from pdf_loader import PDFLoader


loader = PDFLoader("data/AWS.pdf")

documents = loader.load()

print("=" * 50)

print(f"Pages Loaded : {len(documents)}")

print("=" * 50)

print("\nFirst Page\n")

print(documents[0].page_content[:500])

print("\nMetadata\n")

print(documents[0].metadata)
