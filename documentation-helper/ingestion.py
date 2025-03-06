from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import FireCrawlLoader


load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} documents")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_docs)
    
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})
        
    print(f"Going to add {len(documents)} documents to Pinecone")
    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name="langchain-doc-index",
    )
    print("Documents added to Pinecone")
    
    
def ingest_docs2():
    
    langchain_documents_base_urls = [
        "https://python.langchain.com/docs/introduction/"
    ]
    
    for url in langchain_documents_base_urls:
        print(f"FireCrawling {url}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                # "crawlerOptions": {"limit": 5},
                "pageOptions": {"onlyMainContent": True},
                # "wait_until_done": True,
            }
        )
        docs = loader.load()
        
        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name="firecrawl-index",
        )
        print("Documents added to Pinecone")
    
    

if __name__ == "__main__":
    ingest_docs2()
