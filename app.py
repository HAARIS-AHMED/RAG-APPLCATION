
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = str(BASE_DIR / "chroma_db")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL = "openai/gpt-oss-120b:cerebras"

st.set_page_config(
    page_title="Document RAG Chat",
    page_icon="📄"
)


def get_retriever():

    embedding_model = MistralAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


def get_context(docs):

    text = ""

    for doc in docs:
        text += doc.page_content + "\n\n"

    return text


st.title("📄 Document RAG Chat")

if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY is missing from .env")
    st.stop()

if not HF_TOKEN:
    st.error("HF_TOKEN is missing from .env")
    st.stop()

if not os.path.exists(CHROMA_DIR):
    st.error(
        f"Chroma database not found:\n{CHROMA_DIR}"
    )
    st.stop()


try:
    retriever = get_retriever()

    client = InferenceClient(
        api_key=HF_TOKEN
    )

except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query = st.chat_input(
    "Ask something about the document..."
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    with st.chat_message("assistant"):

        try:

            docs = retriever.invoke(query)

            if not docs:

                answer = (
                    "I could not find the answer "
                    "in the document."
                )

                st.markdown(answer)

            else:

                context = get_context(docs)

                prompt = f"""
You are a helpful AI assistant.

Answer the question ONLY using the context below.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{query}
"""

                response = client.chat_completion(
                    model=LLM_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer only from the "
                                "provided document context."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=512,
                    temperature=0.1,
                    stream=False
                )

                answer = response.choices[0].message.content

                st.markdown(answer)

                st.caption(
                    "Sources: " +
                    ", ".join(
                        sorted(
                            {
                                d.metadata.get(
                                    "source",
                                    "unknown"
                                )
                                for d in docs
                            }
                        )
                    )
                )

        except Exception as e:

            st.error(
                f"Error generating answer:\n\n{e}"
            )

            answer = f"Error: {e}"


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
