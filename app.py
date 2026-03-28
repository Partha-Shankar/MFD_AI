import gradio as gr
from main_detector import analyze_media

def run(file):
    analyze_media(file.name)
    return "Analysis Complete"

demo = gr.Interface(
    fn=run,
    inputs=gr.File(),
    outputs="text",
    title="Multimodal Fake Content Detector"
)

demo.launch()