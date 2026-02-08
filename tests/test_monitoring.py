"""
Test per moduli monitoring.
Gestisce gracefully l'assenza di Evidently AI.
"""

import pytest
import pandas as pd
import os


def test_data_quality_import():
    """Test import modulo data quality."""
    try:
        from src.monitoring.data_quality import (
            create_data_quality_report,
            generate_data_quality_report,
            EVIDENTLY_AVAILABLE,
        )
        
        # Verifica che il modulo sia importabile
        assert True
    except ImportError as e:
        pytest.fail(f"Errore import data_quality: {e}")


def test_data_drift_import():
    """Test import modulo data drift."""
    try:
        from src.monitoring.data_drift import (
            create_data_drift_report,
            check_data_drift,
            EVIDENTLY_AVAILABLE,
        )
        
        assert True
    except ImportError as e:
        pytest.fail(f"Errore import data_drift: {e}")


def test_prediction_drift_import():
    """Test import modulo prediction drift."""
    try:
        from src.monitoring.prediction_drift import (
            create_prediction_drift_report,
            EVIDENTLY_AVAILABLE,
        )
        
        assert True
    except ImportError as e:
        pytest.fail(f"Errore import prediction_drift: {e}")


def test_data_quality_without_evidently():
    """Test data quality quando Evidently non è disponibile."""
    from src.monitoring.data_quality import (
        create_data_quality_report,
        EVIDENTLY_AVAILABLE,
    )
    
    df = pd.DataFrame({
        "text": ["Test 1", "Test 2", "Test 3"],
        "label": ["positive", "negative", "neutral"],
    })
    
    if not EVIDENTLY_AVAILABLE:
        # Se Evidently non è disponibile, dovrebbe sollevare ImportError
        with pytest.raises(ImportError):
            create_data_quality_report(
                reference_data=df,
                current_data=df.head(2),
            )
    else:
        # Se disponibile, esegui test normale
        report = create_data_quality_report(
            reference_data=df,
            current_data=df.head(2),
        )
        assert report is not None


def test_data_drift_without_evidently():
    """Test data drift quando Evidently non è disponibile."""
    from src.monitoring.data_drift import (
        create_data_drift_report,
        EVIDENTLY_AVAILABLE,
    )
    
    df = pd.DataFrame({
        "text": ["Test 1", "Test 2", "Test 3"],
        "label": ["positive", "negative", "neutral"],
    })
    
    if not EVIDENTLY_AVAILABLE:
        with pytest.raises(ImportError):
            create_data_drift_report(
                reference_data=df,
                current_data=df.head(2),
            )
    else:
        # Usa più campioni per evitare errore "After pruning, no terms remain"
        # Evidently richiede almeno ~30 campioni per data drift
        # Crea dataset più grande con più varietà
        large_df = pd.DataFrame({
            "text": [
                "Questo è un testo positivo fantastico",
                "Questo è un testo negativo terribile",
                "Questo è un testo neutro normale",
                "Ottimo prodotto da consigliare",
                "Pessimo servizio da evitare",
                "Servizio standard senza problemi",
            ] * 20,  # 120 campioni totali
            "label": ["positive", "negative", "neutral", "positive", "negative", "neutral"] * 20,
        })
        
        report, drift_results = create_data_drift_report(
            reference_data=large_df.head(60),  # 60 campioni reference
            current_data=large_df.tail(50),    # 50 campioni current
        )
        assert report is not None
        assert isinstance(drift_results, dict)

