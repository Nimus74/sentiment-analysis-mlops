"""
Modulo per split riproducibile train/val/test.
Garantisce split identici per entrambi i modelli.
"""

import os
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml


def stratified_split(
    df: pd.DataFrame,
    label_column: str = "label",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Crea split stratificato train/val/test.
    
    Args:
        df: DataFrame da splittare
        label_column: Nome colonna con le etichette
        train_size: Proporzione training set
        val_size: Proporzione validation set
        test_size: Proporzione test set
        random_seed: Seed per riproducibilità
        stratify: Se True, mantiene distribuzione classi
    
    Returns:
        Tuple (train_df, val_df, test_df, split_indices_dict)
    """
    # Verifica che le proporzioni sommino a 1
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Le proporzioni devono sommare a 1.0, ottenuto: {total}"
        )
    
    # Trova colonna label se non specificata
    if label_column not in df.columns:
        for col in ["sentiment", "sentiment_label", "target"]:
            if col in df.columns:
                label_column = col
                break
        else:
            raise ValueError("Colonna label non trovata")
    
    # Prepara label per stratificazione
    y = df[label_column].values
    
    # Primo split: train vs (val + test)
    train_ratio = train_size
    temp_ratio = val_size + test_size
    
    if stratify:
        train_idx, temp_idx = train_test_split(
            df.index,
            test_size=temp_ratio,
            random_state=random_seed,
            stratify=y,
        )
    else:
        train_idx, temp_idx = train_test_split(
            df.index,
            test_size=temp_ratio,
            random_state=random_seed,
        )
    
    # Secondo split: val vs test
    temp_df = df.loc[temp_idx]
    temp_y = temp_df[label_column].values
    
    val_ratio = val_size / temp_ratio
    
    if stratify:
        val_idx, test_idx = train_test_split(
            temp_df.index,
            test_size=(1 - val_ratio),
            random_state=random_seed,
            stratify=temp_y,
        )
    else:
        val_idx, test_idx = train_test_split(
            temp_df.index,
            test_size=(1 - val_ratio),
            random_state=random_seed,
        )
    
    # Crea DataFrame split
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    
    # Salva indici per tracciabilità
    split_indices = {
        "train": train_idx.values,
        "val": val_idx.values,
        "test": test_idx.values,
    }
    
    return train_df, val_df, test_df, split_indices


def save_split_indices(
    split_indices: Dict[str, np.ndarray],
    output_dir: str,
    metadata: Dict[str, Any],
) -> None:
    """
    Salva indici split con metadata per tracciabilità.
    
    Args:
        split_indices: Dizionario con indici per ogni split
        output_dir: Directory dove salvare
        metadata: Metadata aggiuntivi (seed, proporzioni, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva indici come pickle
    indices_path = os.path.join(output_dir, "split_indices.pkl")
    with open(indices_path, "wb") as f:
        pickle.dump(split_indices, f)
    
    # Salva metadata come JSON
    metadata_path = os.path.join(output_dir, "split_metadata.json")
    metadata["indices_file"] = indices_path
    
    # Converti numpy arrays in liste per JSON
    metadata["split_sizes"] = {
        k: len(v) for k, v in split_indices.items()
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Split indici salvati: {indices_path}")
    print(f"Split metadata salvato: {metadata_path}")


def load_split_indices(indices_path: str) -> Dict[str, np.ndarray]:
    """
    Carica indici split salvati.
    
    Args:
        indices_path: Path al file pickle con gli indici
    
    Returns:
        Dizionario con indici per ogni split
    """
    with open(indices_path, "rb") as f:
        split_indices = pickle.load(f)
    return split_indices


def apply_split_indices(
    df: pd.DataFrame,
    split_indices: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applica indici split salvati a un DataFrame.
    
    Args:
        df: DataFrame da splittare
        split_indices: Dizionario con indici per ogni split
    
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    train_df = df.loc[split_indices["train"]].reset_index(drop=True)
    val_df = df.loc[split_indices["val"]].reset_index(drop=True)
    test_df = df.loc[split_indices["test"]].reset_index(drop=True)
    
    return train_df, val_df, test_df


def verify_split_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = "label",
) -> Dict[str, Any]:
    """
    Verifica che la distribuzione delle classi sia mantenuta negli split.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        label_column: Nome colonna label
    
    Returns:
        Dizionario con statistiche distribuzione
    """
    train_dist = train_df[label_column].value_counts(normalize=True)
    val_dist = val_df[label_column].value_counts(normalize=True)
    test_dist = test_df[label_column].value_counts(normalize=True)
    
    # Calcola deviazioni
    deviations = {}
    for label in train_dist.index:
        train_pct = train_dist[label]
        val_pct = val_dist.get(label, 0)
        test_pct = test_dist.get(label, 0)
        
        deviations[label] = {
            "train": float(train_pct),
            "val": float(val_pct),
            "test": float(test_pct),
            "max_deviation": float(max(abs(val_pct - train_pct), abs(test_pct - train_pct))),
        }
    
    return {
        "train_distribution": train_dist.to_dict(),
        "val_distribution": val_dist.to_dict(),
        "test_distribution": test_dist.to_dict(),
        "deviations": deviations,
    }


def main():
    """
    Funzione principale per creare split.
    Legge configurazione da config.yaml.
    """
    # Carica configurazione
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print("File config.yaml non trovato!")
        return
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    split_config = config.get("split", {})
    paths_config = config.get("paths", {})
    
    # Parametri split
    train_size = split_config.get("train_size", 0.70)
    val_size = split_config.get("val_size", 0.15)
    test_size = split_config.get("test_size", 0.15)
    random_seed = split_config.get("random_seed", 42)
    stratify = split_config.get("stratify", True)
    
    # Paths
    data_processed = paths_config.get("data_processed", "data/processed")
    splits_dir = paths_config.get("data_splits", "data/splits")
    
    # Carica dataset processato
    dataset_path = os.path.join(data_processed, "dataset_processed.csv")
    if not os.path.exists(dataset_path):
        print(f"Dataset processato non trovato: {dataset_path}")
        print("Esegui prima il preprocessing!")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"Dataset caricato: {len(df)} campioni")
    
    # Crea split
    train_df, val_df, test_df, split_indices = stratified_split(
        df=df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_seed=random_seed,
        stratify=stratify,
    )
    
    print(f"\nSplit creati:")
    print(f"  Train: {len(train_df)} campioni ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} campioni ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} campioni ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verifica distribuzione
    distribution = verify_split_distribution(train_df, val_df, test_df)
    print("\nDistribuzione classi mantenuta:")
    for label, stats in distribution["deviations"].items():
        print(f"  {label}: max deviation = {stats['max_deviation']:.4f}")
    
    # Salva split
    train_df.to_csv(os.path.join(data_processed, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_processed, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_processed, "test.csv"), index=False)
    
    # Salva indici
    metadata = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "random_seed": random_seed,
        "stratify": stratify,
        "total_samples": len(df),
    }
    
    save_split_indices(split_indices, splits_dir, metadata)
    
    print(f"\n✅ Split salvati in: {data_processed}")


if __name__ == "__main__":
    main()

