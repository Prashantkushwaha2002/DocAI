import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

DB_FAISS_PATH = "vectorstore/db_faiss"

# Load vector store only once
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# Load LLM only once
@st.cache_resource
def load_llm():
    return pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=300
    )

def main():
    st.set_page_config(page_title="Medical RAG Chatbot")
    st.title("Medical RAG Chatbot")

    try:
        vectorstore = load_vectorstore()
    except Exception as e:
        st.error(f"Vectorstore load failed: {e}")
        return

    llm_pipeline = load_llm()

    query = st.text_input("Ask your medical question:")

    if query:
        try:
            # Step 1: Retrieve relevant chunks
            docs = vectorstore.similarity_search(query, k=3)

            if not docs:
                st.write("No relevant documents found.")
                return

            # Step 2: Combine context
            context = "\n\n".join([doc.page_content for doc in docs])

            # Step 3: Prompt
            prompt = f"""
You are a medical assistant.

Answer ONLY using the provided context.
If the answer is not found in the context, say:
"Insufficient information."

Context:
{context}

Question:
{query}

Answer:
"""

            # Step 4: Generate response
            response = llm_pipeline(prompt)[0]["generated_text"]

            st.write("### Answer:")
            st.write(response)

            # Optional: Show sources
            with st.expander("Show Source Chunks"):
                for i, doc in enumerate(docs):
                    st.write(f"Source {i+1}:")
                    st.write(doc.page_content[:500])
                    st.write("---")

        except Exception as e:
            st.error(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
