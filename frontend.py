import streamlit as st
import requests
from uuid import uuid4


st.header("DocMind QA Web App")

BACKEND_URL = "https://docmind-rag-production-d5e8.up.railway.app"

if "session_id" not in st.session_state:
    st.session_state["session_id"]=str(uuid4())


if "uploaded" not in st.session_state:
    st.session_state.uploaded=False

if "messages" not in st.session_state:
    st.session_state.messages=[]

if not st.session_state.uploaded:
    file=st.file_uploader("Upload a PDF File",type=['pdf'])
    if file is not None:
        if st.button("Upload"):
                file_data={"file":(file.name,file.getvalue(),"application/pdf")}
                response=requests.post(f"{BACKEND_URL}/upload",files=file_data,params={"session_id":st.session_state["session_id"]})
                
                if response.status_code==200:
                    st.session_state.uploaded=True
                    st.success("File upload Sucessfully!")
                    st.rerun()
                else:
                    st.error("Upload failed. Try again.")

else:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("📄 Document ready Ask your questions below")
    with col2:
        if st.button("New Document"):
            st.session_state.uploaded = False
            st.session_state.messages = []
            st.session_state["session_id"]=str(uuid4())
            st.rerun()

if st.session_state.uploaded:
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    question=st.chat_input("Ask a question about the document...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("assistant"):
            response=requests.post(f"{BACKEND_URL}/ask",json={"question":question,"session_id":st.session_state["session_id"]})
            data=response.json()
            answer = data["answer"]
            sources = data["sources"]
            st.write(answer)
            if sources and "I don't know" not in answer:
                all_pages = []
                for page in sources:
                    page_number=f"Page {page}"
                    all_pages.append(page_number)
                new_pages=", ".join(all_pages)
                st.write(f"Sources:{new_pages}")
            
            st.session_state.messages.append({"role":"assistant","content":answer})
