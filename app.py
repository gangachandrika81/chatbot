import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

import langchain_groq

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ======================================================
# LOAD ENV
# ======================================================
load_dotenv()

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
    color: white;
}

h1 {
    text-align: center;
    color: #38bdf8;
}

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stSidebar"] {
            
    background-color: #111827;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    background-color: #2563eb;
    color: white;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================
st.title("🤖 AI RAG Chatbot")

st.markdown(
    """
    <center>
    Chat normally or upload PDFs for AI-powered RAG answers.
    </center>
    """,
    unsafe_allow_html=True
)

# ======================================================
# SESSION STATE
# ======================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:

    st.header("📄 Upload PDF Files")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.vector_db = None
        st.rerun()

    st.markdown("---")

    st.success("✅ Streaming Responses")
    st.success("✅ RAG PDF Chat")
    st.success("✅ Multiple PDFs")
    st.success("✅ Chat Memory")

# ======================================================
# DISPLAY CHAT HISTORY
# ======================================================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ======================================================
# PROCESS PDFs
# ======================================================
if uploaded_files and st.session_state.vector_db is None:

    with st.spinner("📚 Processing PDFs..."):

        all_docs = []

        for uploaded_file in uploaded_files:

            # Skip empty files
            if uploaded_file.size == 0:
                st.warning(f"{uploaded_file.name} is empty.")
                continue

            # Save temp PDF
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())
                temp_pdf_path = tmp_file.name

            try:

                # Load PDF
                loader = PyPDFLoader(temp_pdf_path)

                docs = loader.load()

                all_docs.extend(docs)

            except Exception as e:

                st.error(f"Error reading {uploaded_file.name}")
                st.error(e)

        # No documents found
        if len(all_docs) == 0:

            st.error("❌ No readable text found in PDFs.")
            st.stop()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )

        documents = splitter.split_documents(all_docs)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create FAISS DB
        vector_db = FAISS.from_documents(
            documents,
            embeddings
        )

        # Save in session
        st.session_state.vector_db = vector_db

        st.success("✅ PDFs Processed Successfully")

# ======================================================
# USER INPUT
# ======================================================
question = st.chat_input("Ask something...")

# ======================================================
# MAIN CHATBOT
# ======================================================
if question:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    try:

        # ======================================================
        # LOAD MODEL
        # ======================================================
        llm = langchain_groq.ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            streaming=True
        )

        # ======================================================
        # CREATE CHAT HISTORY
        # ======================================================
        chat_history = []

        for msg in st.session_state.messages:                                           

            if msg["role"] == "user":

                chat_history.append(
                    HumanMessage(content=msg["content"])
                )

            else:

                chat_history.append(
                    AIMessage(content=msg["content"])
                )

        # ======================================================
        # ASSISTANT RESPONSE
        # ======================================================
        with st.chat_message("assistant"):

            # ======================================================
            # RAG MODE
            # ======================================================
            if st.session_state.vector_db:

                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": 4}
                )

                retrieved_docs = retriever.invoke(question)

                context = "\n\n".join([
                    doc.page_content
                    for doc in retrieved_docs
                ])

                # Prompt
                prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Previous Conversation:
{history}

PDF Context:
{context}

User Question:
{question}

Answer clearly and accurately.
""")

                # Chain
                chain = (
                    prompt
                    | llm
                    | StrOutputParser()
                )

                # Stream response
                stream = chain.stream({
                    "history": str(chat_history),
                    "context": context,
                    "question": question
                })

                response = st.write_stream(stream)

            # ======================================================
            # NORMAL CHAT MODE
            # ======================================================
            else:

                stream = llm.stream(chat_history)

                response = st.write_stream(
                    chunk.content
                    for chunk in stream
                )

        # ======================================================
        # SAVE AI RESPONSE
        # ======================================================
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    except Exception as e:

        st.error(f"❌ Error: {str(e)}")