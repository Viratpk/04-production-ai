import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(persist_directory="database", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# Rewrite Follow-up Questions
# -----------------------------

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Given the chat history and the latest user question,
rewrite the question so it can be understood
without the chat history.

Do NOT answer the question.
Only rewrite it if needed.
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

# -----------------------------
# Final QA Prompt
# -----------------------------

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not found,
say you don't know.

Context:

{context}
""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)

history = InMemoryChatMessageHistory()

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

while True:
    question = input("\nAsk > ")

    if question.lower() == "exit":
        break

    response = conversational_rag.invoke(
        {
            "input": question,
        },
        config={"configurable": {"session_id": "virat"}},
    )

    print("\n=========================")
    print(response["answer"])
    print("=========================")
