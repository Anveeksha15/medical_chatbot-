import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

DB_FAISS_PATH = "vectorstores/db_faiss"

# ---------------- PROMPT ---------------- #

custom_prompt = """You are a knowledgeable medical assistant.
Use the provided context to answer the user's question accurately and concisely.
If the context does not contain the answer, respond with:
"I'm sorry, I don't have that information in my current knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=custom_prompt,
    input_variables=["context", "question"]
)

# ---------------- LLM ---------------- #

@st.cache_resource
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in .env file.")
        st.stop()
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=512,
        api_key=api_key
    )

@st.cache_resource
def load_retriever():
    if not os.path.exists(DB_FAISS_PATH):
        st.error("⚠️ Vector store not found. Please run `python ingest.py` first.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db.as_retriever(search_kwargs={"k": 3})

# ---------------- APP ---------------- #

st.set_page_config(page_title="Medical Assistant", page_icon="🩺")
st.title("🩺 Medical Assistant")
st.caption("Powered by Gale Encyclopedia of Medicine")

llm = load_llm()
retriever = load_retriever()

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

query = st.chat_input("Ask a medical question...")

if query:
    st.session_state.chat.append(("user", query))

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical knowledge base..."):
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt_text = prompt.format(context=context, question=query)

            response = llm.invoke(prompt_text)
            answer = response.content  # ChatGroq returns an object, not a string

        st.markdown(answer)

    st.session_state.chat.append(("assistant", answer))