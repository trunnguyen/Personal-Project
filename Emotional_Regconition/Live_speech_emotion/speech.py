import gradio as gr
import sounddevice as sd
import queue
import json
import threading
from vosk import Model, KaldiRecognizer
from datetime import datetime

MODEL_PATH = "vosk-model-small-en-us-0.15"

model = Model(MODEL_PATH)
samplerate = 16000

q = queue.Queue()
running = False
transcript_text = ""

def callback(indata, frames, time, status):
    if running:
        q.put(bytes(indata))

def listen_loop():
    global transcript_text, running

    rec = KaldiRecognizer(model, samplerate)

    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback
    ):
        while running:
            data = q.get()

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    transcript_text += text + " "
            else:
                partial = json.loads(rec.PartialResult())
                yield transcript_text + partial.get("partial", "")

def start():
    global running
    running = True
    return "🟢 Listening..."

def stop():
    global running
    running = False
    return "🔴 Stopped"

def save_text():
    global transcript_text

    if not transcript_text.strip():
        return "Nothing to save"

    filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    return f"Saved to {filename}"

def clear_text():
    global transcript_text
    transcript_text = ""
    return ""

with gr.Blocks() as demo:
    gr.Markdown("# 🎤 Offline Live Speech-to-Text (Save to File)")

    status = gr.Textbox(value="Idle", label="Status")
    output = gr.Textbox(label="Live Transcript", lines=15)

    with gr.Row():
        start_btn = gr.Button("Start")
        stop_btn = gr.Button("Stop")
        save_btn = gr.Button("💾 Save to TXT")
        clear_btn = gr.Button("Clear")

    start_btn.click(start, outputs=status)
    stop_btn.click(stop, outputs=status)

    start_btn.click(listen_loop, outputs=output)
    save_btn.click(save_text, outputs=status)
    clear_btn.click(clear_text, outputs=output)

demo.launch()
