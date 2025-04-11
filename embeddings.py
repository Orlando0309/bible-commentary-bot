from langchain_community.embeddings.ollama import OllamaEmbeddings
def get_embedding_function():
    embedings = OllamaEmbeddings(
        model="nomic-embed-text",
    )
    return embedings