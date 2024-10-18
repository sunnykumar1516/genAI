from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def loadAndcreateEmbeddings():
    loader = PyPDFLoader("rawData/sample.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text = text_splitter.split_documents(docs)
    embeddings = (
        OllamaEmbeddings(model = "gemma2:2b")
    )
    db = FAISS.from_documents(text,embeddings)
    
    query = "Moreover, Augmented Reality/Virtual Reality/ Mixed Reality (AR/VR/MR)"
    db.save_local("db/FAISS/index")
    def queryIt():
        res = db.similarity_search(query)
        print(res[0])
        print("------------------------------")
    def queryItasRetriver():
        retriver = db.as_retriever()
        res = retriver.invoke(query)
        print(res[0])
    def loadLocal():
        new_db = FAISS.load_local("db/FAISS/index",
                                  embeddings,
                                  allow_dangerous_deserialization=True)
        res = new_db.similarity_search(query)
        print(res[0])
    #queryIt()
    #queryItasRetriver()
    
    loadLocal()
loadAndcreateEmbeddings()