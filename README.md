# 📄 Document RAG Chatbot

An AI-powered **Document Retrieval-Augmented Generation (RAG)** chatbot that answers questions using information from an ingested document.

The application uses **Mistral AI embeddings** to convert document content into vectors, **ChromaDB** for similarity-based retrieval, and a **Hugging Face-hosted LLM** to generate answers from the retrieved context.

## 🚀 Features

* Ask questions about the ingested document
* Retrieval using ChromaDB
* MMR-based document retrieval
* Mistral AI embeddings
* Hugging Face-hosted LLM
* Source references for retrieved content
* Conversational follow-up questions
* Streamlit web interface
* Download conversation transcript

## 🛠️ Technologies Used

* Python
* Streamlit
* LangChain
* Mistral AI
* ChromaDB
* Hugging Face Inference API

## 📁 Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
└── chroma_db/
    ├── chroma.sqlite3
    └── 9510d6c6-a84c-4fd4-a0ee-b1f04a277253/
        ├── data_level0.bin
        ├── header.bin
        ├── index_metadata.pickle
        ├── length.bin
        └── link_lists.bin
```

## 🔑 Environment Variables

The application requires two API keys:

```text
MISTRAL_API_KEY
HF_TOKEN
```

For Hugging Face Spaces, add these through:

**Settings → Secrets**

Do not upload your `.env` file or expose your API keys publicly.

## ▶️ How It Works

```text
Document
   ↓
Text Chunking
   ↓
Mistral Embeddings
   ↓
ChromaDB
   ↓
User Question
   ↓
Relevant Chunks Retrieved
   ↓
Hugging Face LLM
   ↓
Context-based Answer
```

## 💬 Example

**Question:**

```text
What is deep learning?
```

The chatbot retrieves the relevant sections from the document and generates an answer based only on the retrieved context.

If the information is not available in the document, the chatbot responds:

```text
I could not find the answer in the document.
```

## 🌐 Running Locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## ⚠️ Important

The current application uses an already-created **ChromaDB vector store**. New documents must be processed and embedded before their content can be searched by the chatbot.
