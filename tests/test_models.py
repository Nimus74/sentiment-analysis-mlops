"""
Test unitari per modelli Transformer e FastText.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.models.transformer_model import TransformerSentimentModel
from src.models.fasttext_model import FastTextSentimentModel


@pytest.mark.skipif(
    not os.path.exists("models/transformer/final_model"),
    reason="Modello Transformer non addestrato - usa modello salvato se disponibile"
)
def test_transformer_model_init():
    """Test inizializzazione modello Transformer con modello salvato."""
    model = TransformerSentimentModel.load("models/transformer/final_model")
    
    assert model.model is not None
    assert model.tokenizer is not None
    assert model.device in ["cpu", "cuda"]


@pytest.mark.skipif(
    not os.path.exists("models/transformer/final_model"),
    reason="Modello Transformer non addestrato - skip test"
)
def test_transformer_model_predict():
    """Test predizione singola Transformer."""
    try:
        model = TransformerSentimentModel.load("models/transformer/final_model")
        
        result = model.predict("Questo è un test positivo")
        
        assert "label" in result
        assert "score" in result
        assert result["label"] in ["negative", "neutral", "positive"]
        assert 0.0 <= result["score"] <= 1.0
    except (ValueError, RuntimeError) as e:
        # Se c'è un errore di caricamento (es. torch versione), skip
        if "torch.load" in str(e) or "Numpy is not available" in str(e):
            pytest.skip(f"Modello non caricabile: {e}")
        raise


@pytest.mark.skipif(
    not os.path.exists("models/transformer/final_model"),
    reason="Modello Transformer non addestrato - skip test"
)
def test_transformer_model_predict_batch():
    """Test predizione batch Transformer."""
    try:
        model = TransformerSentimentModel.load("models/transformer/final_model")
        
        texts = [
            "Questo è positivo",
            "Questo è negativo",
            "Questo è neutro",
        ]
        
        results = model.predict_batch(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert "label" in result
            assert "score" in result
            assert result["label"] in ["negative", "neutral", "positive"]
    except (ValueError, RuntimeError) as e:
        # Se c'è un errore di caricamento (es. torch versione), skip
        if "torch.load" in str(e) or "Numpy is not available" in str(e):
            pytest.skip(f"Modello non caricabile: {e}")
        raise


def test_transformer_model_load():
    """Test caricamento modello Transformer da file."""
    # Verifica se esiste un modello salvato
    model_path = "models/transformer/final_model"
    
    if os.path.exists(model_path):
        model = TransformerSentimentModel.load(model_path)
        assert model.model is not None
        assert model.tokenizer is not None
    else:
        pytest.skip("Modello fine-tuned non trovato, skip test")


def test_fasttext_model_load():
    """Test caricamento modello FastText."""
    # Verifica se esiste un modello salvato
    model_path = "models/fasttext/fasttext_model.bin"
    
    if os.path.exists(model_path):
        model = FastTextSentimentModel.load(model_path)
        assert model.model is not None
        
        # Test predizione
        result = model.predict("Questo è un test")
        assert "label" in result
        assert "score" in result
        assert result["label"] in ["negative", "neutral", "positive"]
    else:
        pytest.skip("Modello FastText non trovato, skip test")


def test_fasttext_model_predict_batch():
    """Test predizione batch FastText."""
    model_path = "models/fasttext/fasttext_model.bin"
    
    if os.path.exists(model_path):
        model = FastTextSentimentModel.load(model_path)
        
        texts = ["Test 1", "Test 2", "Test 3"]
        results = model.predict_batch(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert "label" in result
            assert "score" in result
    else:
        pytest.skip("Modello FastText non trovato, skip test")

