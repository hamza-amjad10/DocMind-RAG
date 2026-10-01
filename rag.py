from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pickle

load_dotenv()


# embedding model
embedding_model=HuggingFaceEndpointEmbeddings( model="BAAI/bge-small-en-v1.5")

# generation model
model=ChatGroq(model="openai/gpt-oss-20b")

# prompt
prompt=PromptTemplate(
    template=""" Answer the question using ONLY the information in the context below.
    The context may describe the answer without repeating the exact term used in the question
    if a matching definition or explanation is present, use it to answer.
    If the information is genuinely not present in the context, say "I don't know based on the provided document."
    Do not use any outside knowledge beyond what's in the context.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:""",
    input_variables=["context", "query"]
)

#string parser
parser = StrOutputParser()


def ingest_pdf(file_path: str):
    
    loader = PyPDFLoader(file_path)
    text = loader.load()
    
    # All pages text combine to make single text
    full_text = "\n\n".join([doc.page_content for doc in text if doc.page_content.strip()])
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n●", "\n○", "\n", " ", ""]
    )
    
    chunks = splitter.create_documents([full_text])
    
    old_store = Chroma(embedding_function=embedding_model, persist_directory="chroma_db")
    old_store.delete_collection()
    
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    
    bm_retriever=BM25Retriever.from_documents(
        documents=chunks
    )
    bm_retriever.k = 8
    
    with open("bm25_retriever.pkl", "wb") as f:
            pickle.dump(bm_retriever, f)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )
    
    return vectorstore
    


def ask_question(query:str):
    
    docs=[]
    
    with open("bm25_retriever.pkl", "rb") as f:
        bm_retriever = pickle.load(f)
    
    
    vectorstore=Chroma(
        embedding_function=embedding_model,
        persist_directory="chroma_db"
    )

    vector_retriever =vectorstore.as_retriever(search_kwargs={"k": 8})
    
    ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm_retriever],
    weights=[0.5, 0.5]  # 50% vector, 50% keyword
    )
    
    result_docs = ensemble_retriever.invoke(query)

    for doc in result_docs:
        docs.append(doc.page_content)


    string_text="\n\n".join(docs)


    chain=prompt|model|parser

    result=chain.invoke({"context":string_text,"query":query})
    return result



if __name__ == "__main__":
    print("Starting ingestion...")
    ingest_pdf("company_handbook.pdf")
    print("Ingestion done, asking question...")
    answer = ask_question("Can I carry over unused annual leave?")
    print("Final answer:", answer)