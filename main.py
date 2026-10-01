from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel
from rag import ask_question,ingest_pdf
import os
import shutil


app=FastAPI()

class Query(BaseModel):
    question: str


@app.post('/upload')
def upload_file(file: UploadFile=File(...)):
    os.makedirs("uploads",exist_ok=True)
    
    save_path=f"uploads/{file.filename}"
    with open(save_path,"wb") as f:
        shutil.copyfileobj(file.file,f)
    
    ingest_pdf(save_path)
    return {"filename": file.filename, "status": "uploaded and processed"}
    


@app.post('/ask')
def ask_query(data: Query):
    try:
        answer = ask_question(data.question)
    except FileNotFoundError:
        answer = "Please upload a PDF first."
    return {"answer":answer}
    
