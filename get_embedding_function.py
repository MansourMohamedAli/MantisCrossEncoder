# from langchain_community.embeddings.ollama import OllamaEmbeddings # deprecated
from langchain_ollama import OllamaEmbeddings
# from nomic import embed
# import numpy as np

def get_embedding_function(model, base_url):
    # embeddings = OllamaEmbeddings(model="llama3.2",
    #                               base_url='http://127.0.0.1:11434',
    #                               show_progress=True,
    #                               num_ctx=20000,
    #                               num_thread=1)
    embeddings = OllamaEmbeddings(model=model,
                                  base_url=base_url)
    print(len(embeddings.get_query_embedding("1")))
    # output = embed(model=model, base_url=base_url)
    # output = embed.text(model=model, base_url=base_url)
    # embeddings = np.array(output['embeddings'])
    return embeddings
    
if __name__ == "__main__":
    print(get_embedding_function("granite3-dense:8b"))
