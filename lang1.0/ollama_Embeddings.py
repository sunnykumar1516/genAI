from langchain_community.embeddings import OllamaEmbeddings

embeddings = (
    OllamaEmbeddings(model = "gemma2:2b")
)
text = "my name is sunny, what's your name"
query_result = embeddings.embed_query(text)
print(query_result)