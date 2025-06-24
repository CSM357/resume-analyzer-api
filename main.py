from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from resume_utils import parse_resume
from rag_engine import store_and_embed, ask_question

app = FastAPI()

# Allow Android app to access it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

resume_chunks = []

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    content = await file.read()
    text = parse_resume(content)
    global resume_chunks
    resume_chunks = store_and_embed(text)
    return {"status": "success", "message": "Resume processed."}

@app.post("/ask")
async def ask(query: dict):
    question = query.get("question")
    answer = ask_question(question, resume_chunks)
    return {"answer": answer}
