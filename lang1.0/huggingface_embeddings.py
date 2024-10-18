import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
text = "testing the embeddings by sunny"
query = embeddings.embed_query(text)
print(query)