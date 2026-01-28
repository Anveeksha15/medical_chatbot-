import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers

DB_FAISS_PATH = "../vectorstores/db_faiss"

# ---------------- PROMPT ---------------- #

custom_prompt = """You are a knowledgeable medical assistant.
Use the provided context to answer the user's question accurately and concisely.
If the context does not contain the answer, respond with:
"I'm sorry, I don't have that information."

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
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGUF",
        model_file="llama-2-7b-chat.Q4_K_M.gguf",
        model_type="llama",
        config={
            "context_length": 2048,
            "max_new_tokens": 256,
            "temperature": 0.5
        }
    )

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db.as_retriever(search_kwargs={"k": 2})

llm = load_llm()
retriever = load_retriever()

# ---------------- UI ---------------- #


st.set_page_config(page_title="Medical Assistant", page_icon="🩺")
st.title("🩺 Medical Assistant")

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Display previous messages
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

# User input
query = st.chat_input("Ask a medical question")

if query:
    st.session_state.chat.append(("user", query))

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt_text = prompt.format(
        context=context,
        question=query
    )

    answer = llm.invoke(prompt_text)

    st.session_state.chat.append(("assistant", answer))

    st.rerun()
