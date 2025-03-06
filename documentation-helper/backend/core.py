from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()


INDEX_NAME = "langchain-doc-index"


def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    stuff_documents_chain = create_stuff_documents_chain(
        llm=chat,
        prompt=retrieval_qa_chat_prompt,
    )
    retrieval_qa_chain = create_retrieval_chain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=stuff_documents_chain,
    )
    
    result = retrieval_qa_chain.invoke({"input": query})
    return dict(
        query = result["input"],
        result=result["answer"],
        source_documents=result["context"],
    )


if __name__ == "__main__":
    print(run_llm("What is LangChain Chain?"))
