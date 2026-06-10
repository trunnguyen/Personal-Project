from langchain_community.vectorstores import FAISS
from modules.embeddings import load_embeddings

def create_vector_db(chunks):

    embedding_model = load_embeddings()

    texts = [doc.page_content for doc in chunks]

    vectordb = FAISS.from_texts(texts, embedding_model)

    return vectordb