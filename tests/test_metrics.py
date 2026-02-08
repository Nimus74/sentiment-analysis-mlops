"""
Test unitari per calcolo metriche.
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    calculate_metrics,
    check_metrics_thresholds,
    compare_models_metrics,
)


def test_calculate_metrics():
    """Test calcolo metriche base."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    
    metrics = calculate_metrics(y_true, y_pred, labels=["negative", "neutral", "positive"])
    
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "confusion_matrix" in metrics
    
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_check_metrics_thresholds():
    """Test verifica soglie metriche."""
    metrics = {
        "macro_f1": 0.80,
        "negative_f1": 0.75,
        "neutral_f1": 0.70,
        "positive_f1": 0.85,
    }
    
    thresholds = {
        "macro_f1_min": 0.75,
        "per_class_f1_min": 0.60,
    }
    
    passes, messages = check_metrics_thresholds(metrics, thresholds)
    assert passes is True
    assert len(messages) > 0


def test_compare_models_metrics():
    """Test confronto metriche due modelli."""
    metrics1 = {
        "accuracy": 0.85,
        "macro_f1": 0.82,
        "macro_precision": 0.83,
        "macro_recall": 0.81,
    }
    
    metrics2 = {
        "accuracy": 0.90,
        "macro_f1": 0.88,
        "macro_precision": 0.89,
        "macro_recall": 0.87,
    }
    
    comparison = compare_models_metrics(metrics1, metrics2, "Model1", "Model2")
    
    assert len(comparison) > 0
    assert "Metric" in comparison.columns
    assert "Model1" in comparison.columns
    assert "Model2" in comparison.columns

