from langchain_community.document_loaders import PyPDFLoader

def load_documents(file):

    loader = PyPDFLoader(file.name)
    documents = loader.load()

    return documents