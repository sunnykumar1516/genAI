from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def loadData():
    loader = PyPDFLoader("../rawData/sample.pdf")
    docs = loader.load()
    return docs
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text = text_splitter.split_documents(docs)
    return text

def getdata_splitted_data():
    docs = loadData()
    text = split_text(docs)
    return text