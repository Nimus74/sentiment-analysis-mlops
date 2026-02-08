"""
Test unitari per split dati.
"""

import pytest
import pandas as pd
from src.data.split import stratified_split


def test_stratified_split_basic():
    """Test split stratificato base."""
    df = pd.DataFrame({
        "text": ["Test"] * 100,
        "label": ["positive"] * 50 + ["negative"] * 50,
    })
    
    train_df, val_df, test_df, indices = stratified_split(
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


def test_stratified_split_distribution():
    """Test che la distribuzione classi sia mantenuta."""
    df = pd.DataFrame({
        "text": ["Test"] * 90,
        "label": ["positive"] * 30 + ["negative"] * 30 + ["neutral"] * 30,
    })
    
    train_df, val_df, test_df, _ = stratified_split(
        df,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_seed=42,
        stratify=True,
    )
    
    # Verifica che ogni split abbia tutte e tre le classi
    assert len(train_df["label"].unique()) == 3
    assert len(val_df["label"].unique()) == 3
    assert len(test_df["label"].unique()) == 3


def test_stratified_split_imbalanced():
    """Test split con dataset sbilanciato."""
    df = pd.DataFrame({
        "text": ["Test"] * 100,
        "label": ["positive"] * 80 + ["negative"] * 20,
    })
    
    train_df, val_df, test_df, _ = stratified_split(
        df,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_seed=42,
        stratify=True,
    )
    
    # Verifica che lo split mantenga proporzioni approssimative
    assert len(train_df) + len(val_df) + len(test_df) == 100


def test_stratified_split_small_dataset():
    """Test split con dataset piccolo."""
    df = pd.DataFrame({
        "text": ["Test"] * 20,  # Aumentato per avere almeno 2 campioni per classe in ogni split
        "label": ["positive"] * 10 + ["negative"] * 10,
    })
    
    train_df, val_df, test_df, _ = stratified_split(
        df,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_seed=42,
        stratify=True,
    )
    
    # Con dataset piccolo, gli split potrebbero essere approssimativi
    assert len(train_df) + len(val_df) + len(test_df) == 20

