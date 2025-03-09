import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sentence_transformers import SentenceTransformer, CrossEncoder
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

class InputData(BaseModel):
    sentences: list

@app.get("/")
def root():
   return {"message": "howdy_world"}

@app.post("/predict")
def predict(data: InputData):
    corpus_embeddings = bi_encoder.encode(data.sentences, convert_to_tensor=True, show_progress_bar=True).tolist()
    return {"corpus_embeddings" : corpus_embeddings}

# Run using: uvicorn filename:app --host 0.0.0.0 --port 8000
