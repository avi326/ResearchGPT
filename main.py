import langchain as lc
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
import pinecone
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

# Load API keys from .env file
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gpt4_api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

# Load PDF files
pdf_files = ["./data/files/file1.pdf", "./data/files/file2.pdf", "./data/files/file3.pdf"]

# Extract text from PDF files
pdf_texts = {file:PyPDFLoader(file).load_and_split()[0].page_content for file in pdf_files}

# Create a Pinecone namespace for storing PDF embeddings
pinecone.init(api_key=pinecone_api_key, environment="northamerica-northeast1-gcp")
# check if index already exists (it shouldn't if this is first time)
index_name = "pdf-embeddings"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=512)

# Embed PDF texts using GPT-4
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=gpt4_api_key)
pdf_embeddings = [model.encode(text).tolist() for text in pdf_texts.values()]

# Store PDF embeddings in Pinecone
index = pinecone.Index(index_name=index_name)
index.upsert(vectors=list(zip(pdf_files,pdf_embeddings)))


# Define Chatbot class
class Chatbot:
    def __init__(self, llm, model, pinecone_namespace):
        self.model = model
        self.llm = llm
        self.pinecone_namespace = pinecone_namespace

    def ask(self, question):
        question_embedding = self.model.encode(question).tolist()

        # Search for the most relevant PDF using Pinecone
        results = index.query(vector=question_embedding, top_k=1)
        most_relevant_pdf = results["matches"][0]["id"]

        # Generate a response using GPT-4
        messages = [
            SystemMessage(content=f" question: {question} context: {pdf_texts[most_relevant_pdf]}")
        ]
        response = self.llm(messages)
        return response


# Initialize the chatbot
chatbot = Chatbot(model=model, llm=llm, pinecone_namespace=index_name)

# Example usage
question = "What is the main topic of file1.pdf?"
response = chatbot.ask(question)
print(response.content, end='\n')

# # Cleanup
# pinecone.deinit()
# pinecone.init(api_key="YOUR_PINECONE_API_KEY")
# pinecone.delete_namespace("pdf_embeddings")
# pinecone.deinit()
