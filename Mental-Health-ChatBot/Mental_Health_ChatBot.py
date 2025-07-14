from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gradio as gr
def initialize_llm():
  llm = ChatGroq(
    temperature = 0,
    groq_api_key = "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",
    model_name = "llama-3.3-70b-versatile"
)
  return llm

def create_vector_db():
  loader = DirectoryLoader(r"C:\Users\nguye\OneDrive\Documents\Data Analysis\Mental_health\mental_health_Document.pdf", loader_cls = PyPDFLoader)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  texts = text_splitter.split_documents(documents)
  embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
  vector_db = Chroma.from_documents(texts, embeddings, persist_directory = './chroma_db')
  vector_db.persist()

  print("ChromaDB created and data saved")

  return vector_db

def setup_qa_chain(vector_db, llm):
  retriever = vector_db.as_retriever()
  prompt_templates = """ You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
  PROMPT = PromptTemplate(template = prompt_templates, input_variables = ['context', 'question'])

  qa_chain = RetrievalQA.from_chain_type(
      llm = llm,
      chain_type = "stuff",
      retriever = retriever,
      chain_type_kwargs = {"prompt": PROMPT}
  )
  return qa_chain


print("Intializing Chatbot.........")
llm = initialize_llm()

db_path = "./chroma_db"

if not os.path.exists(db_path):
  vector_db  = create_vector_db()
else:
  embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
  vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
qa_chain = setup_qa_chain(vector_db, llm)

# def chatbot_response(user_input, history = []):
#     if not user_input.strip():
#         return history + [("User", "Please provide a valid input")]
#     response = qa_chain.run(user_input)
#     return history + [(user_input, response)]


# def chatbot_response(user_input, history):
#     if not user_input.strip():
#         history.append(("User", "Please provide a valid input."))
#     else:
#         response = qa_chain.run(user_input)
#         history.append((user_input, response))
#     return history


# def chatbot_response(user_input, history):
#     if not user_input.strip():
#         history.append(("User", "Please provide a valid input."))
#         return history
#
#     response = qa_chain.run(user_input)
#
#     # Ensure response is a string
#     response = str(response)
#
#     # Append as tuple (user_input, response)
#     history.append((user_input, response))
#     return history


def chatbot_response(user_input, history):
    if not user_input.strip():
        return "Please provide a valid input."

    response = qa_chain.run(user_input)
    return str(response)


with gr.Blocks() as app:
    gr.Markdown("# 🧠 Mental Health Chatbot 🤖")
    gr.Markdown("A compassionate chatbot...")

    chatbot = gr.ChatInterface(
        fn=chatbot_response,
        title="Mental Health Chatbot"
    )

    gr.Markdown("This chatbot provides general support...")

app.launch()
