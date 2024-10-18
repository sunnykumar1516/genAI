import data_ingestion as dg
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
def createEmbeddings_and_saveIt():
    docs = dg.getdata_splitted_data()
    embeddings = (
        OllamaEmbeddings(model="gemma2:2b")
    )
    vectorSoreDB = FAISS.from_documents(docs,embeddings)
    vectorSoreDB.save_local("../vectorDB/index")

def get_saved_vector():
    embeddings = (
        OllamaEmbeddings(model="gemma2:2b")
    )
    new_db = FAISS.load_local("../vectorDB/index",
                              embeddings,
                              allow_dangerous_deserialization=True)
    return new_db
 
