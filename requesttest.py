from langchain_community.document_loaders.csv_loader import CSVLoader
import pickle
from pathlib import Path
import requests
import ollama

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

def generate_passages(data_path):
    docs = encode_csv(data_path)
    passages = []
    for doc in docs:
        passages.append(str([doc.metadata['Problem Summary'], doc.metadata['Problem Description'], doc.metadata['Notes & Resolution']]))
    return passages

def generate_embeddings(passages, embedding_name: str):
    root = Path().resolve()
    embedding_cache_path = root / 'embeddings' / embedding_name
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
    return corpus_embeddings

# generate a response combining the prompt and data we retrieved in step 2
query = "Who is dofarrell?"
passages = generate_passages(data_path="data/mantis.csv")
embeddings = generate_embeddings(passages, embedding_name='doc_embedding.pickle')

llm_payload = {"query" : query, "embeddings" : embeddings, "passages" : passages }

rerank_results_json = requests.post("http://127.0.0.1:8000/invoke_llm", json=llm_payload)
rerank_results = rerank_results_json.json()["rerank_results"]

client = ollama.Client(host='http://localhost:11434')

model = "llama3.2:latest"
print(f"\n------------------------ Start {model} Response ------------------------ \n")
output = client.chat(model=model, messages=[{'role': 'user', 'content': f"Using this data: {rerank_results}. Respond to this prompt: {query}"}])
print(output.message.content)
print(f"\n------------------------- End {model} Response ------------------------- \n")