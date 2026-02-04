from fastapi import FastAPI
from pydantic import BaseModel
from loader import build_or_load_vectorstore
from rag_chain import build_rag_chain

app = FastAPI(title="Sabarimala RAG Chatbot")

PDF_PATH = "data/Sabarimala_RAG_Text_Heavy_9_Pages.pdf"

# ---- Load RAG components ONCE (startup) ----
vectorstore = build_or_load_vectorstore(PDF_PATH)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
rag_chain = build_rag_chain(retriever)


# ---- Request schema ----
class ChatRequest(BaseModel):
    question: str


# ---- Response schema ----
class ChatResponse(BaseModel):
    answer: str


# ---- Health check ----
@app.get("/")
def health():
    return {"status": "RAG chatbot is running"}


# ---- Chat endpoint ----
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = rag_chain.invoke(req.question)
    return {"answer": answer}
