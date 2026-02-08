"""
Script helper per eseguire preprocessing e split dati.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import preprocess_dataframe
from src.data.validation import generate_quality_report
from src.data.split import stratified_split, save_split_indices
import yaml


def main():
    """Esegue preprocessing e split completo."""
    # Carica configurazione
    config_path = Path("configs/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    preprocessing_config = config.get("preprocessing", {})
    split_config = config.get("split", {})
    paths_config = config.get("paths", {})
    
    # Paths
    data_raw = paths_config.get("data_raw", "data/raw")
    data_processed = paths_config.get("data_processed", "data/processed")
    data_splits = paths_config.get("data_splits", "data/splits")
    
    os.makedirs(data_processed, exist_ok=True)
    os.makedirs(data_splits, exist_ok=True)
    
    # Carica dataset raw
    dataset_path = os.path.join(data_raw, "dataset.csv")
    print(f"Caricamento dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset caricato: {len(df)} campioni")
    
    # Preprocessing
    print("\n=== PREPROCESSING ===")
    df_processed = preprocess_dataframe(
        df,
        text_column="text",
        remove_urls_flag=preprocessing_config.get("remove_urls", True),
        remove_mentions_flag=preprocessing_config.get("remove_mentions", True),
        normalize_hashtags_flag=preprocessing_config.get("normalize_hashtags", True),
        normalize_special_chars_flag=preprocessing_config.get("normalize_special_chars", True),
        min_length=preprocessing_config.get("min_text_length", 3),
        max_length=preprocessing_config.get("max_text_length", 512),
    )
    
    print(f"Dopo preprocessing: {len(df_processed)} campioni")
    
    # Salva dataset processato
    processed_path = os.path.join(data_processed, "dataset_processed.csv")
    df_processed.to_csv(processed_path, index=False)
    print(f"Dataset processato salvato: {processed_path}")
    
    # Validazione qualità
    print("\n=== VALIDAZIONE QUALITÀ ===")
    report_path = os.path.join(data_processed, "quality_report.json")
    generate_quality_report(
        df_processed,
        report_path,
        text_column="text",
        label_column="label",
    )
    
    # Split
    print("\n=== SPLIT TRAIN/VAL/TEST ===")
    train_df, val_df, test_df, split_indices = stratified_split(
        df_processed,
        train_size=split_config.get("train_size", 0.70),
        val_size=split_config.get("val_size", 0.15),
        test_size=split_config.get("test_size", 0.15),
        random_seed=split_config.get("random_seed", 42),
        stratify=split_config.get("stratify", True),
    )
    
    print(f"Train: {len(train_df)} campioni ({len(train_df)/len(df_processed)*100:.1f}%)")
    print(f"Val: {len(val_df)} campioni ({len(val_df)/len(df_processed)*100:.1f}%)")
    print(f"Test: {len(test_df)} campioni ({len(test_df)/len(df_processed)*100:.1f}%)")
    
    # Salva split
    train_df.to_csv(os.path.join(data_processed, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_processed, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_processed, "test.csv"), index=False)
    
    # Salva indici split
    metadata = {
        "train_size": split_config.get("train_size", 0.70),
        "val_size": split_config.get("val_size", 0.15),
        "test_size": split_config.get("test_size", 0.15),
        "random_seed": split_config.get("random_seed", 42),
        "stratify": split_config.get("stratify", True),
        "total_samples": len(df_processed),
    }
    
    save_split_indices(split_indices, data_splits, metadata)
    
    print("\n✅ Preprocessing e split completati!")
    print(f"   File salvati in: {data_processed}")


if __name__ == "__main__":
    main()

