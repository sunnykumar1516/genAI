from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
from langchain_openai import ChatOpenAI
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"


prompt = ChatPromptTemplate.from_template(
    '''
    answer the following question based on below context:
    <context>
    {context}
    <context>
    '''
)

prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system","respond like a scientist"),
        ("user","Question:{question}")
    ]
)


llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt2|llm|output_parser

# strealit
st.title("testing initial models")
input_text = st.text_input("Ask a question")

if input_text:
    st.write(chain.invoke({"question": input_text}))

