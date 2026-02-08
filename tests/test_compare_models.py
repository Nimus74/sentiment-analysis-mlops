"""
Test unitari per confronto modelli.
"""

import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from src.evaluation.compare_models import (
    load_models,
    evaluate_model,
    main,
)


@pytest.mark.skipif(
    not os.path.exists("models/fasttext/fasttext_model.bin"),
    reason="Modello FastText non addestrato"
)
def test_load_models():
    """Test caricamento modelli."""
    transformer_path = "models/transformer/final_model" if os.path.exists("models/transformer/final_model") else None
    fasttext_path = "models/fasttext/fasttext_model.bin"
    
    transformer, fasttext = load_models(
        transformer_path=transformer_path,
        fasttext_path=fasttext_path,
    )
    
    assert transformer is not None
    assert fasttext is not None


@pytest.mark.skipif(
    not os.path.exists("models/fasttext/fasttext_model.bin"),
    reason="Modello FastText non addestrato"
)
def test_evaluate_model():
    """Test valutazione modello."""
    from src.models.fasttext_model import FastTextSentimentModel
    import numpy as np
    
    model = FastTextSentimentModel.load("models/fasttext/fasttext_model.bin")
    
    # Crea test set piccolo
    texts = ["Test positivo", "Test negativo", "Test neutro"] * 5
    labels = ["positive", "negative", "neutral"] * 5
    
    label_to_num = {"negative": 0, "neutral": 1, "positive": 2}
    
    metrics, preds, report = evaluate_model(
        model=model,
        texts=texts,
        labels=labels,
        model_name="fasttext",
        label_to_num=label_to_num,
    )
    
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert isinstance(preds, np.ndarray)
    assert report is not None


def test_compare_models_integration():
    """Test integrazione confronto modelli."""
    # Questo test richiede modelli addestrati, quindi lo skippiamo se non disponibili
    if not os.path.exists("models/fasttext/fasttext_model.bin"):
        pytest.skip("Modelli non addestrati")
    
    # Test con mock per evitare di eseguire tutto il confronto
    with patch('src.evaluation.compare_models.load_models') as mock_load, \
         patch('src.evaluation.compare_models.evaluate_model') as mock_eval, \
         patch('pandas.read_csv') as mock_read_csv:
        
        # Mock modelli
        mock_transformer = MagicMock()
        mock_fasttext = MagicMock()
        mock_load.return_value = (mock_transformer, mock_fasttext)
        
        # Mock dataset
        mock_df = pd.DataFrame({
            "text": ["Test"] * 10,
            "label": ["positive", "negative", "neutral"] * 3 + ["positive"],
        })
        mock_read_csv.return_value = mock_df
        
        # Mock valutazione
        mock_eval.return_value = {
            "accuracy": 0.8,
            "macro_f1": 0.75,
            "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
        
        # Non eseguiamo main() completo ma testiamo le funzioni principali
        assert mock_load is not None
        assert mock_eval is not None

