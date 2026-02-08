"""
Test unitari per preprocessing.
"""

import pytest
import pandas as pd
from src.data.preprocessing import (
    remove_urls,
    remove_mentions,
    normalize_hashtags,
    normalize_special_chars,
    clean_text,
    preprocess_dataframe,
)


def test_remove_urls():
    """Test rimozione URL."""
    text = "Check this out: https://example.com"
    result = remove_urls(text)
    assert "https://example.com" not in result
    assert "Check this out:" in result


def test_remove_mentions():
    """Test rimozione menzioni."""
    text = "Hey @user123, how are you?"
    result = remove_mentions(text)
    assert "@user123" not in result
    assert "Hey" in result


def test_normalize_hashtags():
    """Test normalizzazione hashtag."""
    text = "This is #awesome"
    result = normalize_hashtags(text)
    assert "#" not in result
    assert "awesome" in result


def test_normalize_special_chars():
    """Test normalizzazione caratteri speciali."""
    text = "Test &amp; example"
    result = normalize_special_chars(text)
    assert "&amp;" not in result
    assert "&" in result


def test_clean_text():
    """Test funzione completa di pulizia."""
    text = "Check @user https://example.com #awesome &amp; test"
    result = clean_text(text)
    assert "@user" not in result
    assert "https://example.com" not in result
    assert "#" not in result
    assert "&amp;" not in result


def test_preprocess_dataframe():
    """Test preprocessing DataFrame."""
    df = pd.DataFrame({
        "text": [
            "Test @user https://example.com",
            "Another #test",
            "Short",
        ],
        "label": ["positive", "negative", "neutral"],
    })
    
    result = preprocess_dataframe(df, min_length=5)
    
    # Verifica che testi corti siano rimossi
    assert len(result) == 2
    assert "@user" not in result["text"].iloc[0]
    assert "https://example.com" not in result["text"].iloc[0]

