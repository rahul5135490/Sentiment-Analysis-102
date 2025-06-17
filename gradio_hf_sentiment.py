# gradio_hf_sentiment.py

import gradio as gr
from transformers import pipeline

# Load Hugging Face pipeline
classifier = pipeline("sentiment-analysis")

# Define prediction function
def predict_sentiment(text):
    result = classifier(text)[0]  # {'label': 'POSITIVE', 'score': 0.999...}
    label = result["label"]
    score = round(result["score"], 3)
    return f"Sentiment: {label} (Confidence: {score})"

# Create Gradio UI
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence..."),
    outputs="text",
    title="🤗 Hugging Face Sentiment Analysis",
    description="Enter any English sentence to analyze its sentiment using Hugging Face Transformers."
)

# Launch the app
demo.launch()
