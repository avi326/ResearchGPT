from langchain.embeddings import SentenceTransformerEmbeddings

def init_model():
    model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return model
