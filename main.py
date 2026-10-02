from fastapi import FastAPI,UploadFile,File,HTTPException
from pydantic import BaseModel
from rag import ask_question,ingest_pdf
import os
import shutil
import re


app=FastAPI()

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

def validate_session(session_id:str):
    if not UUID_RE.match(session_id):
        raise HTTPException(status_code=400,detail="Invalid session id")

class Query(BaseModel):
    question: str
    session_id: str


@app.post('/upload')
def upload_file(session_id:str,file: UploadFile=File(...)):
    validate_session(session_id)
    if file.content_type!="application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    os.makedirs(f"uploads/{session_id}",exist_ok=True)
    
    save_path=f"uploads/{session_id}/doc.pdf"
    with open(save_path,"wb") as f:
        shutil.copyfileobj(file.file,f)
    
    ingest_pdf(save_path,session_id)
    return {"filename": file.filename, "status": "uploaded and processed"}
    

@app.post('/ask')
def ask_query(data: Query):
    validate_session(data.session_id)
    try:
        answer = ask_question(data.question,data.session_id)
    except FileNotFoundError:
        answer={
        "answer": "Please upload a PDF first.",
        "sources": []
        }
    return answer
    
