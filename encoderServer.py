import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sentence_transformers import SentenceTransformer, CrossEncoder, util
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

class InputData(BaseModel):
    sentences: list

class Search(BaseModel):
    query: str
    embeddings : list
    passages : list

@app.get("/")
def root():
   return {"message": "hi"}

@app.post("/biencoder")
def biencoder(data: InputData):
    corpus_embeddings = bi_encoder.encode(data.sentences, convert_to_tensor=True, show_progress_bar=True).tolist()
    return {"corpus_embeddings" : corpus_embeddings}

@app.post("/invoke_llm")
def invoke_llm(input: Search):
    question_embedding = bi_encoder.encode(input.query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()

    hits = util.semantic_search(question_embedding, input.embeddings, top_k=5)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[input.query, input.passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-N Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:100]:
        print("\t{:.3f}\t{}".format(hit['score'], input.passages[hit['corpus_id']].replace("\n", " ")))

    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-N Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    result = list()
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], input.passages[hit['corpus_id']].replace("\n", " ")))
        result.append(input.passages[hit['corpus_id']].replace("\n", " "))
    return({"rerank_results" : result})


# Run using: uvicorn filename:app --host 0.0.0.0 --port 8000
