import ingestion as ig

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# this functions creates embddings of the text
# and save it in the form of vector db.
def createEmbeddings_and_saveIt():
    docs = ig.getdata_splitted_data()
    embeddings = (
        OllamaEmbeddings(model="gemma2:2b") # use this model for embeddings
    )
    vectorSoreDB = FAISS.from_documents(docs,embeddings)
    vectorSoreDB.save_local("VectorDB/index")

def get_Saved_Vector():
    embeddings = (
        OllamaEmbeddings(model="gemma2:2b")
    )
    new_db = FAISS.load_local("vectorDB/index",
                              embeddings,
                              allow_dangerous_deserialization=True)
    return new_db
createEmbeddings_and_saveIt()