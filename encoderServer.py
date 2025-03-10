import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sentence_transformers import SentenceTransformer, CrossEncoder
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

class InputData(BaseModel):
    sentences: list

class Query(BaseModel):
    sentences: str

@app.get("/")
def root():
   return {"message": "hi"}

@app.post("/biencoder")
def biencoder(data: InputData):
    corpus_embeddings = bi_encoder.encode(data.sentences, convert_to_tensor=True, show_progress_bar=True).tolist()
    return {"corpus_embeddings" : corpus_embeddings}

@app.post("/crossencoder")
def crossencoder(data: InputData):
    cross_encoder_scores = cross_encoder.predict(data).tolist()
    return {"cross_encoder_scores" : cross_encoder_scores}

@app.post("/q_embed")
def q_embed(data: Query):
    question_embedding = bi_encoder.encode(data.query, convert_to_tensor=True).tolist()
    question_embedding = question_embedding.cuda()
    return {"question_embedding" : question_embedding}

# Run using: uvicorn filename:app --host 0.0.0.0 --port 8000
