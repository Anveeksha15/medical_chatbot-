import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Path relative to project root
PDF_PATH = "data/Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf"
DB_FAISS_PATH = "vectorstores/db_faiss"

def create_vector_db():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages")
    print(f"Sample text: {documents[0].page_content[:200]}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = text_splitter.split_documents(documents)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

    print(f"Created {len(chunks)} valid chunks")

    if not chunks:
        raise ValueError("No chunks created. Check the PDF content.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)

    print("✅ FAISS vector store saved to:", DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()