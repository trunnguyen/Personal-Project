from pathlib import Path
import sys

PROJECT_ROOT= str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from modules.llm import load_llm
from modules.document_loader import load_documents
from modules.text_splitter import split_documents
from modules.vector_store import create_vector_db

def _tokens(text):
    stopwords = {
        "a", "an", "the", "is", "it", "in", "on", "at", "of", "and", "or",
        "to", "for", "with", "this", "that", "what", "how", "why", "when",
        "are", "was", "were", "be", "been", "do", "does", "did", "have",
        "has", "had", "from", "by", "as", "not", "but",
    }
    return {
        w.lower().strip(".,!?\"'")
        for w in text.split()
        if w.lower().strip(".,!?\"'") not in stopwords
    }

def rag_pipeline(file, query):

    documents = load_documents(file)
    chunks = split_documents(documents)
    vectordb = create_vector_db(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    try:
        scored_docs = vectordb.similarity_search_with_score(query, k=3)
        source_docs = [doc for doc, _ in scored_docs]
        retrieval_scores = [float(score) for _, score in scored_docs]
    except Exception:
        source_docs = retriever.get_relevant_documents(query)
        retrieval_scores = []

    q_tokens = _tokens(query)
    s_tokens = _tokens(" ".join(doc.page_content for doc in source_docs))
    qt_in_sources = len(q_tokens & s_tokens) / len(q_tokens) if q_tokens else 0.0
    avg_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0

    print("\n── Retrieval Accuracy ───────────────────────")
    print(f"  Docs retrieved       : {len(source_docs)}")
    print(f"  Avg similarity score : {avg_score:.4f}")
    print(f"  Query terms in chunks: {qt_in_sources * 100:.2f}%")
    print("─────────────────────────────────────────────\n")


    llm = load_llm()

    prompt = PromptTemplate.from_template(
        "Use the context below to answer the question.\n\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    answer = qa_chain.invoke(query)
    source_text = "\n\n".join([doc.page_content[:200] for doc in source_docs])
    final_output = f"{answer}\n\n---\nSources:\n{source_text}"

    return final_output