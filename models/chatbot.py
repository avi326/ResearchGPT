from langchain.schema import HumanMessage, SystemMessage

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
