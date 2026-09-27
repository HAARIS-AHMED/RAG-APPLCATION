
from dotenv import load_dotenv
import os
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient
load_dotenv()
# -------------------------------
# 1. Embedding Model
# -------------------------------
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)
# -------------------------------
# 2. Load Chroma Vector Database
# -------------------------------
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)
#-------------------------------
# 3. Retriever
# -------------------------------
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
# -------------------------------
# 4. Hugging Face Hosted LLM
# -------------------------------
client = InferenceClient(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
# -------------------------------
# 5. RAG System
# -------------------------------
print("Rag system created")
print("press 0 to exit")
while True:
    query = input("You : ")
    if query == "0":
        break
    # -------------------------------
    # Retrieve relevant documents
    # -------------------------------
    docs = retriever.invoke(query)
    # ------------------------------
    # Create context
    # -------------------------------
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    # -------------------------------
    # Create messages
    # -------------------------------
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."

Do not use outside knowledge."""
        },
        {
            "role": "user",
            "content": f"""Context:

{context}


Question:

{query}"""
        }
    ]
    # -------------------------------
    # Generate answer
    # -------------------------------
    response = client.chat_completion(
        model="openai/gpt-oss-120b",
        messages=messages,
        max_tokens=256,
        temperature=0.1
    )
    # -------------------------------
    # Print answer
    # -------------------------------
    answer = response.choices[0].message.content
    print(f"\nAI: {answer}")

