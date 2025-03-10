from langchain_community.document_loaders.csv_loader import CSVLoader
from sentence_transformers import util
import sys
import csv
import pickle
from pathlib import Path
import requests
import ollama

# maxInt = sys.maxsize

# while True:
#     # decrease the maxInt value by factor 10 
#     # as long as the OverflowError occurs.
#     try:
#         csv.field_size_limit(maxInt)
#         break
#     except OverflowError:
#         maxInt = int(maxInt/10)

# def encoder_api(url: str, data):
#     requests.post(url, json=data)


def encode_csv(path: str):
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
embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)


if not embedding_cache_path.exists():
    corpus_sentences = ...
    print("Encoding the corpus. This might take a while")

    data = {"sentences" : passages}
    corpus_embeddings_request = requests.post("http://127.0.0.1:8000/biencoder", json=data)
    corpus_embeddings = corpus_embeddings_request.json()["corpus_embeddings"]

    print("Storing file on disc")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)

else:
    print("Loading pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']


# This function will search all wikipedia articles for passages that
# answer the query
def search(query):
    print("Input question:", query)

    ##### Semantic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    data = {"query" : query}
    question_embedding = requests.post("http://127.0.0.1:8000/q_embed", json=data)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    data = {"cross_input" : cross_inp}
    cross_scores = requests.post("http://127.0.0.1:8000/crossencoder", json=data)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-N Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:100]:
        print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " ")))

    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-N Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    result = list()
    for hit in hits[0:5]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " ")))
        result.append(passages[hit['corpus_id']].replace("\n", " "))
    return(result)

# generate a response combining the prompt and data we retrieved in step 2
query = "I inserted a malfunction and the overcurrent trip did not occur when it was supposed to?"
data = search(query=query)
client = ollama.Client(host='http://localhost:11434')

model = "llama3.2:latest"
print(f"\n------------------------ Start {model} Response ------------------------ \n")
output = client.chat(model=model, messages=[{'role': 'user', 'content': f"Using this data: {data}. Respond to this prompt: {query}"}])
print(output.message.content)
print(f"\n------------------------- End {model} Response ------------------------- \n")