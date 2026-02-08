"""
Test unitari per modelli Transformer e FastText.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from src.models.transformer_model import TransformerSentimentModel
# FastTextSentimentModel viene importato nel test per evitare problemi di cache


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
        # Forza reload completo del modulo per evitare problemi di cache
        import sys
        import importlib
        import importlib.util
        
        # Rimuovi TUTTI i moduli correlati dalla cache (più aggressivo)
        modules_to_remove = []
        for k in list(sys.modules.keys()):
            if 'fasttext' in k.lower() or k.startswith('src.models'):
                modules_to_remove.append(k)
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Forza reload del modulo dal file system usando importlib.util
        spec = importlib.util.spec_from_file_location(
            "fasttext_model_reload",
            os.path.join(os.path.dirname(__file__), "..", "src", "models", "fasttext_model.py")
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            FastTextSentimentModel = module.FastTextSentimentModel
        else:
            # Fallback: import normale
            from src.models.fasttext_model import FastTextSentimentModel
        
        model = FastTextSentimentModel.load(model_path)
        assert model.model is not None
        
        # Test predizione (FastText restituisce lista)
        result_list = model.predict("Questo è un test")
        
        # Verifica che sia una lista
        assert isinstance(result_list, list), f"Expected list, got {type(result_list)}: {result_list}"
        assert len(result_list) > 0
        
        # Verifica struttura primo elemento
        result = result_list[0]
        assert isinstance(result, dict), f"Expected dict, got {type(result)}: {result}"
        assert "label" in result
        assert "score" in result
        # Verifica che NON ci sia il campo "text" (non presente nel nuovo formato)
        assert "text" not in result, f"Field 'text' should not be present in predict() result: {result}"
        assert result["label"] in ["negative", "neutral", "positive"]
    else:
        pytest.skip("Modello FastText non trovato, skip test")


def test_fasttext_model_predict_batch():
    """Test predizione batch FastText."""
    model_path = "models/fasttext/fasttext_model.bin"
    
    if os.path.exists(model_path):
        # Importa modulo (potrebbe essere già importato, ma assicuriamoci)
        import sys
        if 'src.models.fasttext_model' not in sys.modules:
            from src.models.fasttext_model import FastTextSentimentModel
        else:
            # Usa quello già importato
            FastTextSentimentModel = sys.modules['src.models.fasttext_model'].FastTextSentimentModel
        
        model = FastTextSentimentModel.load(model_path)
        
        texts = ["Test 1", "Test 2", "Test 3"]
        results = model.predict_batch(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert "label" in result
            assert "score" in result
    else:
        pytest.skip("Modello FastText non trovato, skip test")

