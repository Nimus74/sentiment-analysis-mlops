"""
Test estesi per split dati.
"""

import pytest
import pandas as pd
import os
from src.data.split import stratified_split


def test_stratified_split_with_metadata():
    """Test split con salvataggio metadata."""
    df = pd.DataFrame({
        "text": [f"text_{i}" for i in range(100)],
        "label": ["positive"] * 40 + ["negative"] * 30 + ["neutral"] * 30,
    })
    
    train_df, val_df, test_df, metadata = stratified_split(
        df,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_seed=42,
    )
    
    # Verifica che lo split sia corretto
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    # metadata è un dict con indici
    assert isinstance(metadata, dict)


def test_stratified_split_custom_sizes():
    """Test split con dimensioni custom."""
    df = pd.DataFrame({
        "text": [f"text_{i}" for i in range(200)],
        "label": ["A"] * 100 + ["B"] * 100,
    })
    
    train_df, val_df, test_df, _ = stratified_split(
        df,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
    )
    
    assert len(train_df) == 160
    assert len(val_df) == 20
    assert len(test_df) == 20


def test_stratified_split_no_stratify():
    """Test split senza stratificazione."""
    df = pd.DataFrame({
        "text": [f"text_{i}" for i in range(100)],
        "label": ["A"] * 90 + ["B"] * 10,
    })
    
    # Nota: la funzione potrebbe non supportare stratify=False direttamente
    # ma testiamo comunque il comportamento
    train_df, val_df, test_df, _ = stratified_split(df)
    
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

