"""
Test unitari per moduli training.
"""

import pytest
import os
import yaml
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_train_fasttext_import():
    """Test import modulo train_fasttext."""
    from src.training import train_fasttext
    assert train_fasttext is not None


def test_train_transformer_import():
    """Test import modulo train_transformer."""
    from src.training import train_transformer
    assert train_transformer is not None


def test_retrain_fasttext_import():
    """Test import modulo retrain_fasttext."""
    from src.training import retrain_fasttext
    assert retrain_fasttext is not None


def test_config_loading():
    """Test caricamento configurazione."""
    config_path = "configs/config.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        assert "transformer" in config
        assert "fasttext" in config
        assert "mlflow" in config
        assert "monitoring" in config


def test_mlflow_utils_import():
    """Test import mlflow_utils."""
    from src.training.mlflow_utils import (
        setup_mlflow,
        log_params,
        log_metrics,
    )
    
    assert setup_mlflow is not None
    assert log_params is not None
    assert log_metrics is not None

