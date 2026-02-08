"""
Pipeline di preprocessing standardizzata per entrambi i modelli.
Garantisce che Transformer e FastText usino lo stesso preprocessing.
"""

import re
from typing import List, Optional
import pandas as pd


def remove_urls(text: str) -> str:
    """
    Rimuove URL da un testo.
    
    Args:
        text: Testo da processare
    
    Returns:
        Testo senza URL
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text)


def remove_mentions(text: str) -> str:
    """
    Rimuove menzioni (@username) da un testo.
    
    Args:
        text: Testo da processare
    
    Returns:
        Testo senza menzioni
    """
    mention_pattern = r'@\w+'
    return re.sub(mention_pattern, '', text)


def normalize_hashtags(text: str) -> str:
    """
    Normalizza hashtag rimuovendo il simbolo # e separando le parole.
    
    Args:
        text: Testo da processare
    
    Returns:
        Testo con hashtag normalizzati
    """
    # Rimuove # e separa camelCase se presente
    text = re.sub(r'#(\w+)', r'\1', text)
    # Separa camelCase (opzionale, può essere troppo aggressivo)
    # text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text


def normalize_special_chars(text: str) -> str:
    """
    Normalizza caratteri speciali comuni nei social media.
    
    Args:
        text: Testo da processare
    
    Returns:
        Testo con caratteri normalizzati
    """
    # Sostituzioni comuni
    replacements = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '…': '...',
        '–': '-',
        '—': '-',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def clean_text(
    text: str,
    remove_urls_flag: bool = True,
    remove_mentions_flag: bool = True,
    normalize_hashtags_flag: bool = True,
    normalize_special_chars_flag: bool = True,
) -> str:
    """
    Applica tutte le operazioni di pulizia del testo.
    
    Args:
        text: Testo da pulire
        remove_urls_flag: Se True, rimuove URL
        remove_mentions_flag: Se True, rimuove menzioni
        normalize_hashtags_flag: Se True, normalizza hashtag
        normalize_special_chars_flag: Se True, normalizza caratteri speciali
    
    Returns:
        Testo pulito
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = str(text).strip()
    
    if remove_urls_flag:
        text = remove_urls(text)
    
    if remove_mentions_flag:
        text = remove_mentions(text)
    
    if normalize_hashtags_flag:
        text = normalize_hashtags(text)
    
    if normalize_special_chars_flag:
        text = normalize_special_chars(text)
    
    # Rimuove spazi multipli
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    remove_urls_flag: bool = True,
    remove_mentions_flag: bool = True,
    normalize_hashtags_flag: bool = True,
    normalize_special_chars_flag: bool = True,
    min_length: int = 3,
    max_length: Optional[int] = None,
) -> pd.DataFrame:
    """
    Preprocessa un intero DataFrame.
    
    Args:
        df: DataFrame da processare
        text_column: Nome della colonna con il testo
        remove_urls_flag: Se True, rimuove URL
        remove_mentions_flag: Se True, rimuove menzioni
        normalize_hashtags_flag: Se True, normalizza hashtag
        normalize_special_chars_flag: Se True, normalizza caratteri speciali
        min_length: Lunghezza minima testo (rimuove testi più corti)
        max_length: Lunghezza massima testo (tronca se più lungo)
    
    Returns:
        DataFrame preprocessato
    """
    df = df.copy()
    
    # Applica pulizia
    df[text_column] = df[text_column].apply(
        lambda x: clean_text(
            x,
            remove_urls_flag=remove_urls_flag,
            remove_mentions_flag=remove_mentions_flag,
            normalize_hashtags_flag=normalize_hashtags_flag,
            normalize_special_chars_flag=normalize_special_chars_flag,
        )
    )
    
    # Filtra per lunghezza minima
    if min_length:
        df = df[df[text_column].str.len() >= min_length]
    
    # Tronca per lunghezza massima
    if max_length:
        df[text_column] = df[text_column].str[:max_length]
    
    # Rimuovi righe vuote
    df = df[df[text_column].str.len() > 0]
    
    return df.reset_index(drop=True)


def prepare_fasttext_format(
    texts: List[str],
    labels: List[str],
    output_path: str,
) -> None:
    """
    Prepara file nel formato richiesto da FastText.
    Formato: __label__<label> <text>
    
    Args:
        texts: Lista di testi
        labels: Lista di etichette corrispondenti
        output_path: Path dove salvare il file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for text, label in zip(texts, labels):
            # FastText richiede formato: __label__<label> <text>
            formatted_line = f"__label__{label} {text}\n"
            f.write(formatted_line)

