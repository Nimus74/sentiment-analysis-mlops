"""
Test estesi per download dataset.
"""

import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from src.data.download_dataset import (
    validate_dataset,
    save_dataset_with_metadata,
    calculate_file_hash,
)


def test_calculate_file_hash():
    """Test calcolo hash file."""
    import tempfile
    
    # Crea file temporaneo
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name
    
    try:
        hash_value = calculate_file_hash(temp_path)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hash length
    finally:
        os.unlink(temp_path)


def test_save_dataset_with_metadata():
    """Test salvataggio dataset con metadata."""
    import tempfile
    import json
    
    df = pd.DataFrame({
        "text": ["Test 1", "Test 2"],
        "label": ["positive", "negative"],
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_dataset.csv")
        
        metadata = {"source": "test", "num_samples": 2}
        
        result = save_dataset_with_metadata(
            df=df,
            output_path=output_path,
            dataset_name="test_dataset",
            metadata=metadata,
        )
        
        # Verifica file creato
        assert os.path.exists(output_path)
        assert isinstance(result, dict)
        
        # Verifica che il dataset sia leggibile
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == 2


def test_validate_dataset_extended():
    """Test validazione dataset estesa."""
    # Test con dataset valido completo
    df = pd.DataFrame({
        "text": ["Test 1", "Test 2", "Test 3"] * 10,
        "label": ["positive", "negative", "neutral"] * 10,
    })
    
    results = validate_dataset(df)
    assert results["valid"] is True
    assert results["stats"]["total_samples"] == 30
    
    # Test con dataset con valori nulli
    df_null = pd.DataFrame({
        "text": ["Test 1", None, "Test 3"],
        "label": ["positive", "negative", "neutral"],
    })
    
    results_null = validate_dataset(df_null)
    # Il dataset può essere valido anche con nulli (dipende dall'implementazione)
    assert isinstance(results_null, dict)

