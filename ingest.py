from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PDF_PATH = r"Gale Encyclopedia of Medicine Vol. 1 (A-B) (1).pdf"

DB_FAISS_PATH = "../vectorstores/db_faiss"

def create_vector_db():
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s)")
    print("Raw text length:", len(documents[0].page_content))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = text_splitter.split_documents(documents)

    # ✅ MUCH lighter filter
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

    print(f"Created {len(chunks)} valid chunks")

    if not chunks:
        raise ValueError("Text exists but chunking failed.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)

    print("✅ FAISS vector store created successfully")

if __name__ == "__main__":
    create_vector_db()
