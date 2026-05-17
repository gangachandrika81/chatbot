# AI RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot developed using Streamlit, LangChain, Groq LLM, FAISS, and HuggingFace Embeddings.

This application allows users to:

- Chat with an AI assistant in real time
- Upload PDF documents
- Ask questions based on uploaded PDFs
- Receive context-aware responses using RAG
- Experience fast streaming responses with an attractive UI

The project combines Large Language Models (LLMs) with vector search to provide accurate and intelligent document-based question answering.

---

# Features

## AI Chat Support
Users can interact with the chatbot normally without uploading documents.

## RAG-Based PDF Question Answering
Upload one or multiple PDF files and ask questions based on document content.

## Streaming Responses
Responses are displayed in real time for a smoother user experience.

## Multiple PDF Upload Support
Supports processing multiple PDF documents simultaneously.

## FAISS Vector Database
Efficient similarity search and retrieval using FAISS.

## HuggingFace Embeddings
Uses sentence-transformer embeddings for semantic document understanding.

## Chat Memory
Maintains conversation history during the session.

## Attractive Streamlit UI
Modern and responsive user interface with sidebar controls.

## Error Handling
Handles:
- Empty PDFs
- Invalid documents
- API errors
- Embedding issues
- Vector database issues

---

# UML Diagrams

## System Architecture Diagram

flowchart TB

    U[User]

    UI[Streamlit UI]

    PDF[PDF Upload]

    LOAD[PyPDF Loader]

    SPLIT[Text Splitter]

    EMBED[HuggingFace Embeddings]

    DB[(FAISS Vector DB)]

    RETRIEVE[Retriever]

    LLM[Groq LLM]

    RESP[AI Response]

    U --> UI

    UI --> PDF

    PDF --> LOAD

    LOAD --> SPLIT

    SPLIT --> EMBED

    EMBED --> DB

    UI --> RETRIEVE

    RETRIEVE --> DB

    DB --> RETRIEVE

    RETRIEVE --> LLM

    UI --> LLM

    LLM --> RESP

    RESP --> UI


---

## RAG Workflow Diagram


flowchart LR

    A[Upload PDF]
    B[Extract Text]
    C[Create Chunks]
    D[Generate Embeddings]
    E[(Store in FAISS)]

    F[User Question]
    G[Similarity Search]
    H[Relevant Context]
    I[Groq LLM]
    J[Final Answer]

    A --> B --> C --> D --> E

    F --> G

    E --> G

    G --> H

    H --> I

    F --> I

    I --> J


---

## Sequence Diagram


sequenceDiagram

    actor User

    participant UI as Streamlit UI
    participant PDF as PDF Loader
    participant DB as FAISS DB
    participant AI as Groq LLM

    User->>UI: Upload PDF

    UI->>PDF: Extract Text

    PDF->>DB: Store Embeddings

    User->>UI: Ask Question

    UI->>DB: Retrieve Relevant Chunks

    DB-->>UI: Return Context

    UI->>AI: Send Prompt + Context

    AI-->>UI: Generate Response

    UI-->>User: Display Answer



---

# Installation

## Clone the Repository

```bash
git clone https://github.com/gangachandrika81/chatbot.git
```

---

## Navigate to the Project Directory

```bash
cd chatbot
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install streamlit
pip install python-dotenv
pip install langchain
pip install langchain-community
pip install langchain-core
pip install langchain-groq
pip install faiss-cpu
pip install sentence-transformers
pip install pypdf
```

---

## Configure Environment Variables

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from:

https://console.groq.com/keys

---

## Run the Application

```bash
streamlit run app.py
```

---

# How RAG Works

The application follows the Retrieval-Augmented Generation (RAG) pipeline:

1. User uploads one or more PDF files
2. PDFs are processed using PyPDF Loader
3. Extracted text is divided into chunks
4. HuggingFace embeddings are generated
5. Embeddings are stored in FAISS vector database
6. User asks a question
7. Relevant document chunks are retrieved using similarity search
8. Retrieved context is sent to the Groq LLM
9. The AI generates a context-aware response

This approach improves response accuracy by grounding the LLM with relevant document information.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Streamlit | Frontend Web Application |
| LangChain | LLM Application Framework |
| Groq API | Large Language Model |
| FAISS | Vector Database |
| HuggingFace Embeddings | Semantic Embeddings |
| Sentence Transformers | Text Embedding Model |
| PyPDF | PDF Processing |
| Python | Backend Development |

---

# Project Structure

```bash
chatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
└── venv/
```

---

# Example Questions

## General Chat
- What is Artificial Intelligence?
- Explain Machine Learning.
- What is Retrieval-Augmented Generation?

## PDF-Based Questions
- Summarize this document
- What are the key points?
- Explain chapter 2
- Give important concepts from the PDF

---

# Author
D. Iswarya
G.Ganga Chandrika

GitHub:
https://github.com/gangachandrika81

---

# License

This project is open-source and available under the MIT License.
