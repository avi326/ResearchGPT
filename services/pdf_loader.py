from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_data_from_pdfs(file_paths):
    loaders = [PyPDFLoader(file) for file in file_paths]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    return split_docs
