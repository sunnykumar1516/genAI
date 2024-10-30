from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.retrieval import create_retrieval_chain
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
#-------------loading api keys-----------------
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
#-------------end--------------------------------------

def get_saved_vector():
    embeddings = (
        OllamaEmbeddings(model="gemma2:2b")
    )
    index_path = os.path.join(os.getcwd(), "VectorDB/index/index.faiss")
    print("he is path:-",index_path)
    absolut_path = "/Volumes/sunny333/genAI/genAI-repo/lang1.0/genAI_App/RAG_CAVE/VectorDB/index"
    new_db = FAISS.load_local(absolut_path,
                              embeddings,
                              allow_dangerous_deserialization=True)
    return new_db


#--------------- prompts----------------------
prompt=ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context:
<context>
{context}
</context>

"""
)


llm = Ollama(model="gemma2:2b")
doc_chain = create_stuff_documents_chain(llm,prompt)

db = get_saved_vector() #get the saved vectors
retriver = db.as_retriever() # create a retriver
ret_chain=create_retrieval_chain(retriver,doc_chain) # create a retriver chain for context

# strealit
st.title("testing initial models")
input_text = st.text_input("Ask a question")

if input_text:
    res= ret_chain.invoke({"input":input_text})
    st.write(res['answer'])