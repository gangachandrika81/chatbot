import streamlit as st
from dotenv import load_dotenv
import os
import tempfile

from langchain_groq import ChatGroq

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("🤖 AI RAG Chatbot")

st.markdown(
    "Chat normally or upload PDFs for RAG-based answers."
)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.header("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.write("✅ Normal Chat Supported")
    st.write("✅ RAG Mode Supported")
    st.write("✅ Multiple PDFs")
    st.write("✅ Chat Memory Enabled")

# ---------------- DISPLAY CHAT HISTORY ----------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ---------------- USER INPUT ----------------
question = st.chat_input("Ask something...")

# ---------------- MAIN LOGIC ----------------
if question:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Display user message
    with st.chat_message("user"):

        st.markdown(question)

    # ---------------- LOADING ----------------
    with st.spinner("Generating response..."):

        try:

            # ---------------- LOAD MODEL ----------------
            llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama-3.3-70b-versatile"
            )

            # =================================================
            # CREATE CHAT HISTORY
            # =================================================
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

            # =================================================
            # RAG MODE
            # =================================================
            if uploaded_files:

                all_docs = []

                # Process uploaded PDFs
                for uploaded_file in uploaded_files:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp_file:

                        tmp_file.write(
                            uploaded_file.read()
                        )

                        temp_pdf_path = tmp_file.name

                    # Load PDF
                    loader = PyPDFLoader(
                        temp_pdf_path
                    )

                    docs = loader.load()

                    all_docs.extend(docs)

                # Check docs
                if not all_docs:

                    st.error(
                        "No readable text found in PDFs."
                    )

                    st.stop()

                # Split documents
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                documents = splitter.split_documents(
                    all_docs
                )

                if not documents:

                    st.error(
                        "Could not process PDFs."
                    )

                    st.stop()

                # ---------------- EMBEDDINGS ----------------
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # ---------------- VECTOR DB ----------------
                db = FAISS.from_documents(
                    documents,
                    embeddings
                )

                retriever = db.as_retriever()

                # Retrieve relevant docs
                retrieved_docs = retriever.invoke(
                    question
                )

                # Context
                context = "\n\n".join([
                    doc.page_content
                    for doc in retrieved_docs
                ])

                # ---------------- PROMPT ----------------
                prompt = ChatPromptTemplate.from_template("""
                You are a helpful AI assistant.

                Previous Conversation:
                {history}

                Context:
                {context}

                User Question:
                {question}

                Give a clear and accurate answer.
                """)

                # ---------------- CHAIN ----------------
                chain = (
                    prompt
                    | llm
                    | StrOutputParser()
                )

                # Generate response
                response = chain.invoke({
                    "history": str(chat_history),
                    "context": context,
                    "question": question
                })

            # =================================================
            # NORMAL CHAT MODE
            # =================================================
            else:

                response = llm.invoke(
                    chat_history
                ).content

            # ---------------- SAVE RESPONSE ----------------
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            # ---------------- DISPLAY RESPONSE ----------------
            with st.chat_message("assistant"):

                st.markdown(response)

        except Exception as e:

            st.error(f"Error: {e}")