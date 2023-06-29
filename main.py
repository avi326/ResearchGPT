import os
from dotenv import load_dotenv
from services.pdf_loader import create_data_from_pdfs
from services.embedding import init_model
from services.pinecone_service import init_pinecone
from models.chatbot import Chatbot
from langchain.chat_models import ChatOpenAI

# Load API keys from .env file
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gpt4_api_key = os.getenv("OPENAI_API_KEY")

pdf_files = ["./data/files/file1.pdf", "./data/files/file2.pdf", "./data/files/file3.pdf"]
pdf_texts = create_data_from_pdfs(pdf_files)

model = init_model()

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
