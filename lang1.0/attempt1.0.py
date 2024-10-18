# loading the data

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("rawData/sample.pdf")
docs = loader.load()



# splittng the text-----------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
text = text_splitter.split_documents(docs)


import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("Open_AI_KEY")



# embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
embeddings = OpenAIEmbeddings(model= "text-embedding-3-large")

db = Chroma.from_documents(text,embeddings)

# query the text from vector db
query = '''Moreover, Augmented Reality/Virtual Reality/ Mixed Reality
 (AR/VR/MR) will allow you to see how the garments might look
 on you in different lighting conditions'''
retrived_results = db.similarity_search(query)
print(retrived_results)