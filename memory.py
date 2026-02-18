from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Path to FAISS index
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Load Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Load FAISS Database
db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# Step 3: Load Lightweight Local LLM
llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=300
)

# Step 4: Take User Input
user_query = input("Write Query Here: ")

# Step 5: Retrieve Relevant Chunks
docs = db.similarity_search(user_query, k=3)

if not docs:
    print("No relevant documents found.")
else:
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use only the following context to answer the question.
If answer not found, say 'Insufficient information'.

Context:
{context}

Question:
{user_query}

Answer:
"""

    # Step 6: Generate Response
    response = llm_pipeline(prompt)[0]["generated_text"]

    print("\nRESULT:\n", response)

    print("\nSOURCE DOCUMENTS:\n")
    for i, doc in enumerate(docs):
        print(f"Source {i+1}:")
        print(doc.page_content[:400])
        print("------")
