
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
data = PyPDFLoader(
    r"C:\Users\DELL\OneDrive\Desktop\GenAI\deeplearning.pdf"
)

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=r"C:\Users\DELL\OneDrive\Desktop\PROJECTS\RAG APPLICATION\chroma_db"
)

print("Chroma database created successfully!")