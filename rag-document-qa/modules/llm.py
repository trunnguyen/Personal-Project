from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=1024,
        temperature=0.0)