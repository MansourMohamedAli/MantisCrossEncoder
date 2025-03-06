from langchain_community.document_loaders.csv_loader import CSVLoader
import sys
import csv
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm.autonotebook import tqdm
import torch
import ollama
import pickle
from pathlib import Path


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

# if not torch.cuda.is_available():
#     print("Warning: No GPU found. Please add GPU to your notebook")

#We use the Bi-Encoder to encode all passages, so that we can use it with semantic search
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
# bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
top_k = 5                             #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

passages = []
for doc in docs:
    # print(doc.metadata['Problem Summary'], doc.metadata['Problem Description'], doc.metadata['Notes & Resolution'])
    passages.append(str([doc.metadata['Problem Summary'], doc.metadata['Problem Description'], doc.metadata['Notes & Resolution']]))

root = Path().resolve()

embedding_cache_path = root / 'embeddings' / 'doc_embedding.pickle'

if not embedding_cache_path.exists():
    # read your corpus etc
    corpus_sentences = ...
    print("Encoding the corpus. This might take a while")
    corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)

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
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

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
