from langchain_community.vectorstores import FAISS
from modules.embeddings import load_embeddings
from langchain_community.vectorstores.utils import DistanceStrategy

def create_vector_db(chunks):

    embedding_model = load_embeddings()

    texts = [doc.page_content for doc in chunks]

    vectordb = FAISS.from_texts(texts, embedding_model,
                                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

    return vectordb