import gradio as gr
from pathlib import Path
import sys

PROJECT_ROOT= str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from modules.rag_pipeline import rag_pipeline

rag_app = gr.Interface(
    fn=rag_pipeline,
    inputs=[
        gr.File(label="Upload PDF", file_types=[".pdf"]),
        gr.Textbox(label="Ask a Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Retrieval-Augmented Document QA System",
    description="Upload a PDF and ask questions about it."
)


if __name__ == "__main__":
    rag_app.launch(share=True)