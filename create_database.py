from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
## from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
##Load PDF
##split into chunks
## Create the embeddings 
## Store the chorma db
from dotenv import load_dotenv
load_dotenv()
loader=PyPDFLoader("C:/Users/91945/Desktop/RAG/document_loader/deeplearning.pdf")
docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks=splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db-new"
)