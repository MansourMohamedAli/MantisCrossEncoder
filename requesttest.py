from langchain_community.document_loaders.csv_loader import CSVLoader
import sys
import csv
from tqdm.autonotebook import tqdm
import torch
import ollama
import pickle
from pathlib import Path
import requests


maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def encode_csv(path):
    # Load PDF documents
    loader = CSVLoader(file_path=path,
        csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['DR#', 'Problem Summary', 'Problem Description', 'Notes & Resolution']},
        metadata_columns=['DR#', 'Problem Summary', 'Problem Description', 'Notes & Resolution'],
        content_columns=['Problem Summary', 'Problem Description', 'Notes & Resolution'],
        encoding='utf-8')
    documents = loader.load()
    return documents

docs = encode_csv("data/mantis.csv")

passages = []
for doc in docs:
    # print(doc.metadata['Problem Summary'], doc.metadata['Problem Description'], doc.metadata['Notes & Resolution'])
    passages.append(str([doc.metadata['Problem Summary'], doc.metadata['Problem Description'], doc.metadata['Notes & Resolution']]))


root = Path().resolve()
embedding_cache_path = root / 'embeddings' / 'doc_embedding.pickle'

if not embedding_cache_path.exists():
    corpus_sentences = ...
    print("Encoding the corpus. This might take a while")

    data = {"sentences" : passages}
    corpus_embeddings = requests.post("http://127.0.0.1:8000/predict", json=data)

    # print(corpus_embeddings.json()['detail'])
    print(corpus_embeddings.json())

    # print("Storing file on disc")
    # print({'sentences': corpus_sentences, 'embeddings': corpus_embeddings})
    # with open(embedding_cache_path, "wb") as fOut:
    #     pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
















# data = {"sentences" : passages}
# response = requests.post("http://127.0.0.1:8000/predict", json=data)




# print(response.json())