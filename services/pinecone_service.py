import pinecone

def init_pinecone(api_key, index_name, dimension=384):
    pinecone.init(api_key=api_key, environment="northamerica-northeast1-gcp")
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension)
    return pinecone.Index(index_name=index_name)
