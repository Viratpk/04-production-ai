import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="database", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer ONLY using the given context.

Context:
{context}

Question:
{question}
""")

chain = prompt | llm

while True:
    question = input("\nAsk > ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    response = chain.invoke({"context": context, "question": question})

    print("\n===================")
    print(response.content)
    print("===================")
