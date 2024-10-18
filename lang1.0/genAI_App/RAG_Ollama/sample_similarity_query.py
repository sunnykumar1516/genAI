import my_ollama_embedding as oe
db = oe.get_saved_vector()

def sampleQuery():
    
    qText = "what is AR"
    res = db.similarity_search(qText)
    print(res[0])
    
sampleQuery()


