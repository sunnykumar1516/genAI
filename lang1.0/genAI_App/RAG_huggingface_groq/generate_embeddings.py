import data_ingestion as dg
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from data_ingestion import getdata_splitted_data as gd
from langchain_community.vectorstores import FAISS
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

def create_token_HF():
    model = "google/gemma-2-2b"
    hf_embed = HuggingFaceEmbeddings(model_name=model)
    text = gd()
    vectorSoreDB = FAISS.from_documents(text, hf_embed)
create_token_HF()