# 🩺 DocAI – Retrieval-Augmented Medical Chatbot (RAG)

DocAI is an AI-powered Medical Question Answering system built using a Retrieval-Augmented Generation (RAG) architecture.  
The system retrieves relevant medical context from a large medical encyclopedia PDF and generates grounded responses using a local Large Language Model (LLM).

This project demonstrates practical implementation of semantic search, vector databases, and LLM-based answer generation.

---

## 🚀 Key Features

- 📄 Large PDF knowledge ingestion (Medical Encyclopedia)
- ✂ Intelligent document chunking
- 🔎 Semantic similarity search using FAISS
- 🧠 Sentence-Transformer embeddings
- 🤖 Context-grounded LLM answer generation (Flan-T5)
- 🌐 Streamlit interactive UI
- ❌ Hallucination reduction using strict context prompting
- 🏗 Modular architecture (Indexing + Retrieval + Generation + UI)

---

## 🧠 Architecture Overview

1. Medical PDF → Text Extraction  
2. Text → Chunking (RecursiveCharacterTextSplitter)  
3. Chunk Embeddings → FAISS Vector Store  
4. User Query → Semantic Retrieval (Top-k similar chunks)  
5. Retrieved Context → Prompt Template → LLM → Final Answer  

---

## 🛠 Tech Stack

- Python
- LangChain (Community)
- FAISS (Vector Database)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- HuggingFace Transformers
- Streamlit
- PyPDF

---

