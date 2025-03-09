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

data = {"sentences" : passages}
response = requests.post("http://127.0.0.1:8000/predict", json=data)

print(response.json())