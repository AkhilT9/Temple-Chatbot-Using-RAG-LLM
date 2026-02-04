import os
import boto3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings

load_dotenv()

VECTOR_DB_PATH = "vectorstore"


def build_or_load_vectorstore(pdf_path: str):

    bedrock_client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        client=bedrock_client
    )

    # ---- Load existing vector store ----
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # ---- Build vector store (first time only) ----
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)

    return vectorstore
