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
import os
import re

load_dotenv()


def preprocess(text):
    return re.findall(r"\w+", text.lower())


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


def ingest_pdf(file_path: str,session_id: str):
    
    loader = PyPDFLoader(file_path)
    text = loader.load()
    
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n●", "\n○", "\n", " ", ""]
    )
    
    for doc in text:
        t = re.sub(r"\s+", " ", doc.page_content)
        t = re.sub(r"\s*([●○])\s*", r"\n\1 ", t)
        doc.page_content = t.strip()
    
    chunks = splitter.split_documents(text)

    
    os.makedirs("chunks", exist_ok=True)
    os.makedirs("bm25", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)
 
    
    with open(f"chunks/{session_id}.pkl", "wb") as f:
        pickle.dump(chunks, f)
    
    
    bm_retriever = BM25Retriever.from_documents(documents=chunks, preprocess_func=preprocess)
    bm_retriever.k = 8
    
    with open(f"bm25/{session_id}.pkl", "wb") as f:
            pickle.dump(bm_retriever, f)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=f"chroma_db/{session_id}"
    )

    return vectorstore
    

def ask_question(query:str,session_id:str):
    
    docs=[]
    sources=[]
    
    with open(f"bm25/{session_id}.pkl", "rb") as f:
        bm_retriever = pickle.load(f)
    
    
    vectorstore=Chroma(
        embedding_function=embedding_model,
        persist_directory=f"chroma_db/{session_id}"
    )

    vector_retriever =vectorstore.as_retriever(search_kwargs={"k": 8})
    
    ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm_retriever],
    weights=[0.5, 0.5]  # 50% vector, 50% keyword
    )
    
    result_docs = ensemble_retriever.invoke(query)

    for doc in result_docs[:5]:
        docs.append(doc.page_content)
        sources.append(doc.metadata["page"] + 1)

    string_text="\n\n".join(docs)
    sources = sorted(set(sources))


    chain=prompt|model|parser

    result=chain.invoke({"context":string_text,"query":query})
    return {
    "answer": result,
    "sources": sources
    }


if __name__ == "__main__":
    print("Starting ingestion...")
    session_id="test"
    ingest_pdf("dl-curriculum.pdf",session_id)
    print("Ingestion done, asking question...")
    answer = ask_question("what is dropout?",session_id)
    print("Final answer:", answer)
