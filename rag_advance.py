from dotenv import load_dotenv
import os
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient


load_dotenv()

CHROMA_DIR = r"C:\Users\DELL\OneDrive\Desktop\PROJECTS\RAG APPLICATION\chroma_db"

EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL = "openai/gpt-oss-120b"

MAX_TOKENS = 256
TEMPERATURE = 0.1

RETRIEVER_K = 4
RETRIEVER_FETCH_K = 10
RETRIEVER_LAMBDA_MULT = 0.5

HISTORY_TURNS_KEPT = 5


SYSTEM_PROMPT = """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

You may use the recent conversation history for follow-up questions.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Do not use outside knowledge."""


def build_retriever():

    embedding_model = MistralAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )

    print("Documents in database:",
          vectorstore._collection.count())

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVER_K,
            "fetch_k": RETRIEVER_FETCH_K,
            "lambda_mult": RETRIEVER_LAMBDA_MULT
        }
    )


def format_context(docs):

    context = ""

    for doc in docs:
        source = doc.metadata.get("source", "unknown")

        context += (
            f"[Source: {source}]\n"
            f"{doc.page_content}\n\n"
        )

    return context


def get_sources(docs):

    sources = set()

    for doc in docs:
        sources.add(
            doc.metadata.get("source", "unknown")
        )

    return ", ".join(sources)


def get_answer(client, messages):

    response = client.chat_completion(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )

    return response.choices[0].message.content


def main():

    if not os.getenv("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY not found in .env")
        return

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    if not hf_token:
        print("HF_TOKEN not found in .env")
        return

    retriever = build_retriever()

    client = InferenceClient(
        api_key=hf_token
    )

    history = []

    print("\nRAG system ready.")
    print("Type 0 to exit.\n")

    while True:

        query = input("You: ").strip()

        if query == "0":
            break

        if not query:
            continue

        try:
            docs = retriever.invoke(query)

        except Exception as e:
            print(f"\nRetrieval error: {e}\n")
            continue

        if not docs:
            print(
                "\nAI: I could not find the answer "
                "in the document.\n"
            )
            continue

        context = format_context(docs)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Context:\n\n{context}\n\n"
                    f"Question:\n\n{query}"
                )
            }
        )

        try:

            answer = get_answer(
                client,
                messages
            )

            print(f"\nAI: {answer}")

            print(
                f"\nSources: {get_sources(docs)}\n"
            )

        except Exception as e:

            print(
                f"\nGeneration error: {e}\n"
            )

            continue

        history.append(
            {
                "role": "user",
                "content": query
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        history = history[
            -(HISTORY_TURNS_KEPT * 2):
        ]


if __name__ == "__main__":
    main()

