import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os

# Load env variables
load_dotenv()

st.set_page_config(page_title="📚 RAG Chatbot", layout="wide")

st.title("📚 Chat with Your PDF (RAG)")
st.write("Upload a PDF and ask questions!")

# Sidebar
st.sidebar.header("⚙️ Settings")
chunk_size = st.sidebar.slider("Chunk Size", 200, 2000, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 50)

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process PDF
if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        # Embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Vector DB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        st.session_state.vectorstore = vectorstore

        st.success("✅ PDF processed successfully!")

# Chat UI
if st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )

    llm = ChatMistralAI(model="mistral-small-2506")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""),
            ("human",
             """Context:
{context}

Question:
{question}
""")
        ]
    )

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # User input
    user_input = st.chat_input("Ask something about your PDF...")

    if user_input:
        # Show user message
        st.chat_message("user").markdown(user_input)

        # Retrieve docs
        docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt
        final_prompt = prompt.invoke({
            "context": context,
            "question": user_input
        })

        # LLM response
        response = llm.invoke(final_prompt)

        # Show AI response
        st.chat_message("assistant").markdown(response.content)

        # Save chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.content
        })

else:
    st.info("📂 Please upload a PDF to start chatting.")