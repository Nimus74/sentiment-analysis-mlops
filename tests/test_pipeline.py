"""
Test integrazione pipeline end-to-end.
"""

import pytest
import pandas as pd
import os
from src.data.preprocessing import preprocess_dataframe
from src.data.validation import validate_dataset_quality
from src.data.split import stratified_split


def test_preprocessing_pipeline():
    """Test pipeline preprocessing."""
    # Crea dataset di test
    df = pd.DataFrame({
        "text": [
            "Test @user https://example.com #awesome",
            "Another test with &amp; special chars",
            "Simple text here",
        ],
        "label": ["positive", "negative", "neutral"],
    })
    
    # Preprocessa
    processed = preprocess_dataframe(df)
    
    assert len(processed) > 0
    assert "text" in processed.columns
    assert "label" in processed.columns


def test_validation_pipeline():
    """Test pipeline validazione."""
    df = pd.DataFrame({
        "text": ["Test 1", "Test 2", "Test 3"] * 10,
        "label": ["positive", "negative", "neutral"] * 10,
    })
    
    validation = validate_dataset_quality(df)
    
    assert "dataset_size" in validation
    assert "class_distribution" in validation
    assert validation["dataset_size"] == 30


def test_split_pipeline():
    """Test pipeline split."""
    df = pd.DataFrame({
        "text": ["Test"] * 100,
        "label": ["positive"] * 50 + ["negative"] * 50,
    })
    
    train_df, val_df, test_df, _ = stratified_split(
        df,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_seed=42,
    )
    
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert len(train_df) + len(val_df) + len(test_df) == 100

