import gradio as gr
import uuid
import os
import requests
import base64

TTS_OUTPUT_DIR = "./tmp"
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)


def process_audio(audio):
    with open(audio, "rb") as f:
        res = requests.post("http://localhost:8000/process_audio", files={"file": f})
        result = res.json()

    audio_data = base64.b64decode(result["audio"])
    with open(f"{TTS_OUTPUT_DIR}/response.wav", "wb") as f:
        f.write(audio_data)
        
    with open(f"{TTS_OUTPUT_DIR}/asr.txt", "w") as f:
        f.write(result['asr_text'])
    with open(f"{TTS_OUTPUT_DIR}/llm.txt", "w") as f:
        f.write(result['llm_text'])

    return f"""
asr_text: {result['asr_text']}
llm_text: {result['llm_text']}
""", f"{TTS_OUTPUT_DIR}/response.wav"


def on_click_metrics():
    res = requests.get("http://localhost:8000/metrics")
    return res.content.decode('utf-8')


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(value="character.png", show_label=False)  # キャラ絵を表示
        with gr.Column(scale=2):
            mic = gr.Audio(sources=["microphone"], type="filepath", label="Mic")
            text_output = gr.Textbox(label="transcription")
            audio_output = gr.Audio(label="audio", autoplay=True)  

            mic.change(fn=process_audio, inputs=[mic], outputs=[text_output, audio_output])
    with gr.Row():
        metrics_button = gr.Button("compute metrics")
        metrics_output = gr.Textbox(label="Metrics", lines=3)
        metrics_button.click(fn=on_click_metrics, inputs=[], outputs=[metrics_output])

    with gr.Row():
        log = gr.Textbox(label="logs", lines=5)

demo.launch()
