from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub


load_dotenv()


INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: list[tuple[str, str]]):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    
    stuff_documents_chain = create_stuff_documents_chain(
        llm=chat,
        prompt=retrieval_qa_chat_prompt,
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    retrieval_qa_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain,
    )
    
    result = retrieval_qa_chain.invoke({"input": query, "chat_history": chat_history})
    return dict(
        query = result["input"],
        result=result["answer"],
        source_documents=result["context"],
    )


if __name__ == "__main__":
    print(run_llm("What is LangChain Chain?"))
