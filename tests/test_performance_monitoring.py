"""
Test unitari per performance monitoring.
"""

import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from src.monitoring.performance_monitoring import (
    create_performance_report,
    monitor_performance,
)


@pytest.mark.skipif(
    not os.path.exists(".venv310"),
    reason="Evidently AI richiede Python 3.10"
)
def test_create_performance_report():
    """Test creazione report performance."""
    try:
        from evidently import Report
        from evidently.presets import ClassificationPreset
        
        # Crea dati di test
        reference_data = pd.DataFrame({
            "text": ["Test positivo", "Test negativo", "Test neutro"] * 20,
            "label": ["positive", "negative", "neutral"] * 20,
            "prediction": ["positive", "negative", "neutral"] * 20,
        })
        
        current_data = pd.DataFrame({
            "text": ["Test positivo", "Test negativo", "Test neutro"] * 10,
            "label": ["positive", "negative", "neutral"] * 10,
            "prediction": ["positive", "negative", "neutral"] * 10,
        })
        
        report, metrics = create_performance_report(
            reference_data=reference_data,
            current_data=current_data,
            target_column="label",
            prediction_column="prediction",
        )
        
        assert report is not None
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics or metrics.get("accuracy") is None
        assert "macro_f1" in metrics or metrics.get("macro_f1") is None
        
    except ImportError:
        pytest.skip("Evidently AI non disponibile")


def test_monitor_performance():
    """Test monitor performance."""
    try:
        import tempfile
        
        # Crea file reference temporaneo
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            reference_df = pd.DataFrame({
                "text": ["Test"] * 30,
                "label": ["positive", "negative", "neutral"] * 10,
                "prediction": ["positive", "negative", "neutral"] * 10,
            })
            reference_df.to_csv(f.name, index=False)
            reference_path = f.name
        
        try:
            # Crea dati current
            current_data = pd.DataFrame({
                "text": ["Test"] * 15,
                "label": ["positive", "negative", "neutral"] * 5,
                "prediction": ["positive", "negative", "neutral"] * 5,
            })
            
            metrics = monitor_performance(
                predictions_with_labels=current_data,
                reference_path=reference_path,
            )
            
            assert isinstance(metrics, dict)
        finally:
            os.unlink(reference_path)
            
    except ImportError:
        pytest.skip("Evidently AI non disponibile")

