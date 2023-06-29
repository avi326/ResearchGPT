import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.schema import HumanMessage, SystemMessage
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

# TODO: PineconeHybridSearchRetriever

# Load API keys from .env file
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gpt4_api_key = os.getenv("OPENAI_API_KEY")

# Initialize SentenceTransformer model
model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def create_data_from_pdfs(file_paths):
    loaders = [PyPDFLoader(file) for file in file_paths]

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    return split_docs


def init_pinecone(api_key, index_name, dimension=384):
    pinecone.init(api_key=api_key, environment="northamerica-northeast1-gcp")
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension)
    return pinecone.Index(index_name=index_name)


class Chatbot:
    def __init__(self, model, llm, pinecone_index):
        self.model = model
        self.llm = llm
        self.pinecone_index = pinecone_index

    def ask(self, question, text_dict):
        question_embedding = self.model.embed_query(question)

        # Search for the most relevant PDF using Pinecone
        results = self.pinecone_index.query(vector=question_embedding, top_k=5)
        most_relevant_ids = [r["id"] for r in results["matches"]]
        text = "\n \n ".join(text_dict[id] for id in most_relevant_ids)

        # Generate a response using GPT-4
        messages = [
            SystemMessage(content=f" question: {question} context: {text}")
        ]
        response = self.llm(messages)
        return response


pdf_files = ["./data/files/file1.pdf", "./data/files/file2.pdf", "./data/files/file3.pdf"]
pdf_texts = create_data_from_pdfs(pdf_files)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=gpt4_api_key)
embeddings_dict = {str(i): model.embed_query(doc.page_content) for i, doc in enumerate(pdf_texts)}
text_dict = {str(i): doc.page_content for i, doc in enumerate(pdf_texts)}


# Split the list into batches of size 1000
batch_size = 1000
dict_items = list(embeddings_dict.items())
vector_batches = [dict_items[i:i+batch_size] for i in range(0, len(dict_items), batch_size)]

index_name = "pdf-embeddings"
index = init_pinecone(pinecone_api_key, index_name)

for batch in vector_batches:
    index.upsert(vectors=batch)

chatbot = Chatbot(model=model, llm=llm, pinecone_index=index)

question = "explaine is Visual question answering"
response = chatbot.ask(question, text_dict)
print(response.content, end='\n')

# Cleanup
index.delete([], delete_all=True, namespace="pdf_embeddings")
