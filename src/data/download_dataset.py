"""
Script per download e validazione dataset da Hugging Face.
Gestisce il download automatico con tracciabilità e checksum.
"""

import os
import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from datasets import load_dataset
import yaml
import requests


def calculate_file_hash(file_path: str) -> str:
    """
    Calcola hash SHA256 di un file.
    
    Args:
        file_path: Path al file
    
    Returns:
        Hash SHA256 in formato hex
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_italian_sentiment_dataset(
    csv_url: str = "https://huggingface.co/datasets/theoracle/Italian.sentiment.analysis/raw/main/formatted_text.csv",
    cache_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Scarica e parsa il dataset italiano di sentiment analysis da Hugging Face.
    
    Args:
        csv_url: URL del CSV da scaricare
        cache_dir: Directory per cache
    
    Returns:
        DataFrame con colonne 'text' e 'label'
    """
    print(f"Scaricando dataset italiano da: {csv_url}")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Scarica CSV
    response = requests.get(csv_url)
    response.raise_for_status()
    
    # Salva temporaneamente
    temp_path = os.path.join(cache_dir, "formatted_text_raw.csv")
    with open(temp_path, "wb") as f:
        f.write(response.content)
    
    # Leggi e parsa CSV
    df_raw = pd.read_csv(temp_path)
    
    # Parsa formato: "Analyze... [ testo ] = label"
    texts = []
    labels = []
    
    for idx, row in df_raw.iterrows():
        formatted_text = str(row.get("formatted_text", ""))
        
        # Estrai testo tra parentesi quadre
        match = re.search(r'\[(.*?)\]', formatted_text)
        if match:
            text = match.group(1).strip()
        else:
            continue
        
        # Estrai label alla fine (positive, neutral, negative)
        label_match = re.search(r'=\s*(positive|neutral|negative)', formatted_text, re.IGNORECASE)
        if label_match:
            label = label_match.group(1).lower()
        else:
            continue
        
        texts.append(text)
        labels.append(label)
    
    # Crea DataFrame
    df = pd.DataFrame({
        "text": texts,
        "label": labels,
    })
    
    print(f"Dataset parsato: {len(df)} campioni")
    print(f"Distribuzione classi:")
    print(df["label"].value_counts())
    
    return df


def download_dataset(
    dataset_name: str,
    language: Optional[str] = None,
    cache_dir: str = "data/raw",
    split: Optional[str] = None,
    csv_url: Optional[str] = None,
) -> pd.DataFrame:
    """
    Scarica dataset da Hugging Face e lo converte in DataFrame.
    
    Args:
        dataset_name: Nome del dataset su Hugging Face
        language: Filtro lingua (opzionale)
        cache_dir: Directory per cache
        split: Split da scaricare (es. 'train', 'test', None per tutti)
        csv_url: URL diretto a CSV (per dataset custom)
    
    Returns:
        DataFrame con i dati
    """
    # Se è fornito URL CSV diretto, usa funzione specifica
    if csv_url:
        return download_italian_sentiment_dataset(csv_url, cache_dir)
    
    print(f"Scaricando dataset: {dataset_name}")
    
    # Crea directory se non esiste
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Carica dataset
        if split:
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        else:
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        
        # Se è un DatasetDict, prendi il train
        if isinstance(dataset, dict):
            if "train" in dataset:
                dataset = dataset["train"]
            else:
                dataset = list(dataset.values())[0]
        
        # Converti in DataFrame
        df = dataset.to_pandas()
        
        # Filtra per lingua se specificato
        if language and "language" in df.columns:
            df = df[df["language"] == language]
            print(f"Filtrati {len(df)} campioni per lingua: {language}")
        
        print(f"Dataset scaricato: {len(df)} campioni")
        return df
        
    except Exception as e:
        print(f"Errore durante il download: {e}")
        raise


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida formato e qualità del dataset.
    
    Args:
        df: DataFrame da validare
    
    Returns:
        Dizionario con risultati validazione
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    # Verifica colonne necessarie
    required_columns = ["text"]
    for col in required_columns:
        if col not in df.columns:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Colonna mancante: {col}")
    
    # Verifica presenza colonna label/sentiment
    if "label" not in df.columns and "sentiment" not in df.columns:
        validation_results["warnings"].append(
            "Nessuna colonna label/sentiment trovata"
        )
    
    # Statistiche
    validation_results["stats"] = {
        "total_samples": int(len(df)),
        "columns": list(df.columns),
        "null_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "duplicates": int(df.duplicated().sum()),
    }
    
    # Verifica valori nulli in colonna testo
    if "text" in df.columns:
        null_texts = df["text"].isnull().sum()
        if null_texts > 0:
            validation_results["warnings"].append(
                f"{null_texts} testi nulli trovati"
            )
    
    return validation_results


def save_dataset_with_metadata(
    df: pd.DataFrame,
    output_path: str,
    dataset_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Salva dataset con metadata e hash per tracciabilità.
    
    Args:
        df: DataFrame da salvare
        output_path: Path di output
        dataset_name: Nome del dataset
        metadata: Metadata aggiuntivi
    
    Returns:
        Dizionario con path file e hash
    """
    # Crea directory se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salva CSV
    df.to_csv(output_path, index=False)
    
    # Calcola hash
    file_hash = calculate_file_hash(output_path)
    
    # Prepara metadata
    dataset_metadata = {
        "dataset_name": dataset_name,
        "file_path": output_path,
        "sha256_hash": file_hash,
        "num_samples": len(df),
        "columns": list(df.columns),
        **(metadata or {}),
    }
    
    # Salva metadata
    metadata_path = output_path.replace(".csv", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print(f"Dataset salvato: {output_path}")
    print(f"Hash SHA256: {file_hash}")
    
    return {
        "file_path": output_path,
        "metadata_path": metadata_path,
        "hash": file_hash,
    }


def main():
    """
    Funzione principale per download dataset.
    Legge configurazione da config.yaml.
    """
    # Carica configurazione
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print("File config.yaml non trovato!")
        return
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_config = config.get("dataset", {})
    paths_config = config.get("paths", {})
    
    dataset_name = dataset_config.get("name", "cardiffnlp/tweet_sentiment_multilingual")
    language = dataset_config.get("language")
    csv_url = dataset_config.get("csv_url")  # URL diretto per dataset italiano
    cache_dir = paths_config.get("data_raw", "data/raw")
    
    # Download dataset
    df = download_dataset(
        dataset_name=dataset_name,
        language=language,
        cache_dir=cache_dir,
        csv_url=csv_url,
    )
    
    # Valida dataset
    validation = validate_dataset(df)
    print("\nRisultati validazione:")
    print(json.dumps(validation, indent=2))
    
    if not validation["valid"]:
        print("⚠️  Validazione fallita! Controlla gli errori.")
        return
    
    # Salva dataset
    output_path = os.path.join(cache_dir, "dataset.csv")
    save_info = save_dataset_with_metadata(
        df=df,
        output_path=output_path,
        dataset_name=dataset_name,
        metadata={"language": language},
    )
    
    print(f"\n✅ Dataset scaricato e salvato con successo!")
    print(f"   File: {save_info['file_path']}")
    print(f"   Metadata: {save_info['metadata_path']}")


if __name__ == "__main__":
    main()

