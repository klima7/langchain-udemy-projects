import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(name='gpt-4o-mini')
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings
    )
    
    query = "what is Pinecone in machine learning?"

    template = """
Use the following pieces of retrieved context to answer the question at the end.
If you don't know the answer, just say that you don't know. Don't make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always include "thanks for asking!" at the end of your answer.

{context}

Question: {question}

Helpful Answer:
""".strip()

    custom_rag_prompt = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res)
