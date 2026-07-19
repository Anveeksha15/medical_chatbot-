# 🩺 Medical RAG Chatbot (Local LLM + FAISS) 

A **local, privacy‑friendly medical question‑answering chatbot** built using **LangChain**, **FAISS**, **CTransformers (GGUF)**, and **Streamlit**.

The chatbot uses **Retrieval‑Augmented Generation (RAG)** to answer medical questions **only from the provided documents**, ensuring grounded and contextual responses.

---

## 🚀 Features

* ✅ Fully **local LLM** (no OpenAI / cloud calls)
* ✅ **FAISS vector search** for fast document retrieval
* ✅ **GGUF / Llama‑2‑7B** via `ctransformers`
* ✅ **Streamlit chat UI** (stable for CPU models)
* ✅ Source‑grounded medical answers
* ✅ Conversation history in UI

---

## 🧠 Architecture Overview

```
User Question
     ↓
FAISS Retriever (Top‑k = 2)
     ↓
Context Injection (Prompt Template)
     ↓
Local GGUF LLM (Llama‑2‑7B)
     ↓
Answer Displayed in Streamlit Chat
```

---

## 📁 Project Structure

```
med_chatbot/
│
├── data/
│   ├── model.py          # Streamlit app (RAG pipeline + UI)
│   ├── ingest.py         # Document ingestion & FAISS indexing
│
├── vectorstores/
│   └── db_faiss/         # FAISS index (generated)
│
├──requirements.txt
├── README.md
└── venv/                 # (optional) virtual environment
```

---

## ⚙️ Tech Stack

* **Python 3.10+**
* **LangChain (new API)**
* **FAISS** – vector database
* **Sentence‑Transformers** – embeddings
* **CTransformers** – local GGUF inference
* **Llama‑2‑7B‑Chat (Q4_K_M)**
* **Streamlit** – chat UI

---

## 📦 Installation

### 1️⃣ Create & activate virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📄 Prepare Documents

1. Place your medical PDFs / text files in the data folder
2. Run the ingestion script to build the FAISS index:

```bash
python ingest.py
```

This creates:

```
vectorstores/db_faiss/
```

---

## ▶️ Run the Application

```bash
streamlit run model.py
```

Open in browser:

```
http://localhost:8501
```

---

## 💬 Example Query

```
fever symptoms
```

The assistant:

* Retrieves relevant medical context
* Generates a concise, grounded answer
* Avoids hallucination if context is missing

---

## 🔒 Medical Safety Note

⚠️ This project is for **educational and research purposes only**.

* The chatbot **does not provide medical diagnosis**
* Responses depend strictly on provided documents
* Always consult a qualified healthcare professional for medical advice

---

## 🛠️ Configuration Highlights

```python
retriever = db.as_retriever(search_kwargs={"k": 2})

CTransformers(
    context_length=2048,
    max_new_tokens=256,
    temperature=0.5
)
```

These settings are optimized for **CPU‑based GGUF models**.

---

## ❌ Known Limitations

* No token streaming
* CPU‑only inference (slow on large prompts)
* Retrieval is not conversation‑aware
* Requires pre‑indexed documents

---

## 🔮 Future Improvements

* 🔹 Streaming token output
* 🔹 PDF upload + live re‑indexing
* 🔹 Conversation‑aware retrieval
* 🔹 Better citation formatting
* 🔹 Medical safety guardrails

---

## 📜 License

This project is released under the **MIT License**.

---

## 🙌 Acknowledgements

* LangChain
* Hugging Face
* Sentence‑Transformers
* FAISS
* Streamlit
* TheBloke (GGUF models)

---



