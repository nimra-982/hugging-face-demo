from transformers import pipeline
import gradio as gr

# ✅ Force PyTorch to avoid TensorFlow/Keras
model = pipeline("summarization", framework="pt")

def predict(prompt):
    summary = model(prompt)[0]["summary_text"]
    return summary

interface = gr.Interface(fn=predict, inputs="textbox", outputs="text")
interface.launch()
