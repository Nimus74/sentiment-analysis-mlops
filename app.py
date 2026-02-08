"""
App Gradio per Hugging Face Spaces.
Interfaccia UI semplice per sentiment analysis.
"""

import gradio as gr
import os
import sys

# Aggiungi src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.models.transformer_model import TransformerSentimentModel
from src.models.fasttext_model import FastTextSentimentModel

# Carica modelli (usa cache se disponibile)
models = {}


def load_models():
    """Carica modelli."""
    global models
    
    try:
        # Transformer
        transformer_path = "models/transformer/final_model"
        if os.path.exists(transformer_path):
            models["transformer"] = TransformerSentimentModel.load(transformer_path)
        else:
            models["transformer"] = TransformerSentimentModel()
    except Exception as e:
        print(f"Errore caricamento Transformer: {e}")
        models["transformer"] = None
    
    try:
        # FastText
        fasttext_path = "models/fasttext/fasttext_model.bin"
        if os.path.exists(fasttext_path):
            models["fasttext"] = FastTextSentimentModel.load(fasttext_path)
        else:
            models["fasttext"] = None
    except Exception as e:
        print(f"Errore caricamento FastText: {e}")
        models["fasttext"] = None


def predict_sentiment(text: str, model_type: str):
    """Predice sentiment di un testo."""
    if not text or not text.strip():
        return "⚠️ Inserisci un testo da analizzare", ""
    
    model = models.get(model_type)
    
    if model is None:
        return f"❌ Modello {model_type} non disponibile", ""
    
    try:
        result = model.predict(text)
        
        # FastText restituisce lista, Transformer restituisce dict
        if isinstance(result, list):
            result = result[0]
        
        label = result["label"]
        confidence = result["score"]
        
        # Emoji per label
        emoji_map = {
            "positive": "😊",
            "neutral": "😐",
            "negative": "😞",
        }
        
        emoji = emoji_map.get(label, "❓")
        
        output = f"{emoji} **Sentiment**: {label.upper()}\n\n"
        output += f"**Confidence**: {confidence:.2%}"
        
        return output, label
        
    except Exception as e:
        return f"❌ Errore: {str(e)}", ""


# Carica modelli all'avvio
load_models()

# Crea interfaccia Gradio
examples = [
    ["Questo prodotto è fantastico! Lo consiglio a tutti."],
    ["Il servizio è stato ok, niente di speciale."],
    ["Terribile esperienza, non lo consiglio affatto."],
    ["Oggi è una bella giornata di sole."],
    ["Non sono sicuro di cosa pensare di questo."],
]

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=[
        gr.Textbox(
            label="Testo da analizzare",
            placeholder="Inserisci qui il testo...",
            lines=3,
        ),
        gr.Radio(
            choices=["transformer", "fasttext"],
            value="transformer",
            label="Modello",
        ),
    ],
    outputs=[
        gr.Markdown(label="Risultato"),
        gr.Textbox(label="Label", visible=False),
    ],
    title="Sentiment Analysis",
    description="Analizza il sentiment di testi in italiano usando modelli Transformer o FastText.",
    examples=examples,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()

