from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class TextRequest(BaseModel):
    text: str

@app.post("/embedding")
def get_embedding(data: TextRequest):
    embedding = model.encode(data.text).tolist()
    return {"embedding": embedding}
