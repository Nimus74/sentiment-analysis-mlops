"""
Modulo per validazione qualità dati.
Implementa controlli su distribuzione classi, lunghezza testi, valori nulli, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


def check_class_distribution(
    df: pd.DataFrame,
    label_column: str = "label",
) -> Dict[str, Any]:
    """
    Verifica distribuzione delle classi.
    
    Args:
        df: DataFrame da analizzare
        label_column: Nome colonna con le etichette
    
    Returns:
        Dizionario con statistiche distribuzione
    """
    if label_column not in df.columns:
        # Prova colonne alternative
        for col in ["sentiment", "sentiment_label", "target"]:
            if col in df.columns:
                label_column = col
                break
        else:
            return {"error": "Colonna label non trovata"}
    
    value_counts = df[label_column].value_counts()
    total = len(df)
    
    distribution = {
        "total_samples": int(total),
        "num_classes": len(value_counts),
        "class_counts": value_counts.to_dict(),
        "class_percentages": (value_counts / total * 100).to_dict(),
        "is_balanced": False,
    }
    
    # Verifica bilanciamento (considera bilanciato se tutte le classi sono
    # entro il 30% della media)
    if len(value_counts) > 0:
        mean_count = value_counts.mean()
        max_deviation = abs(value_counts - mean_count).max() / mean_count
        distribution["is_balanced"] = max_deviation < 0.3
        distribution["max_deviation_percentage"] = float(max_deviation * 100)
    
    return distribution


def check_text_lengths(
    df: pd.DataFrame,
    text_column: str = "text",
) -> Dict[str, float]:
    """
    Analizza lunghezza dei testi.
    
    Args:
        df: DataFrame da analizzare
        text_column: Nome colonna con i testi
    
    Returns:
        Dizionario con statistiche lunghezza
    """
    if text_column not in df.columns:
        return {"error": "Colonna testo non trovata"}
    
    lengths = df[text_column].astype(str).str.len()
    
    stats = {
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "std": float(lengths.std()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "q25": float(lengths.quantile(0.25)),
        "q75": float(lengths.quantile(0.75)),
        "q90": float(lengths.quantile(0.90)),
        "q95": float(lengths.quantile(0.95)),
    }
    
    return stats


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Verifica valori mancanti.
    
    Args:
        df: DataFrame da analizzare
    
    Returns:
        Dizionario con conteggio valori nulli per colonna
    """
    null_counts = df.isnull().sum()
    return null_counts[null_counts > 0].to_dict()


def check_duplicates(
    df: pd.DataFrame,
    text_column: str = "text",
) -> Dict[str, Any]:
    """
    Verifica duplicati nel dataset.
    
    Args:
        df: DataFrame da analizzare
        text_column: Colonna da usare per identificare duplicati
    
    Returns:
        Dizionario con info duplicati
    """
    duplicates = df.duplicated(subset=[text_column], keep=False)
    num_duplicates = duplicates.sum()
    
    return {
        "num_duplicates": int(num_duplicates),
        "percentage": float(num_duplicates / len(df) * 100) if len(df) > 0 else 0.0,
        "duplicate_indices": df[duplicates].index.tolist() if num_duplicates > 0 else [],
    }


def validate_dataset_quality(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
) -> Dict[str, Any]:
    """
    Esegue validazione completa qualità dataset.
    
    Args:
        df: DataFrame da validare
        text_column: Nome colonna testo
        label_column: Nome colonna label
    
    Returns:
        Dizionario con tutti i risultati validazione
    """
    validation_results = {
        "dataset_size": len(df),
        "class_distribution": check_class_distribution(df, label_column),
        "text_length_stats": check_text_lengths(df, text_column),
        "missing_values": check_missing_values(df),
        "duplicates": check_duplicates(df, text_column),
        "warnings": [],
        "errors": [],
    }
    
    # Verifica errori critici
    if validation_results["missing_values"]:
        validation_results["warnings"].append(
            f"Valori mancanti trovati: {validation_results['missing_values']}"
        )
    
    if validation_results["duplicates"]["num_duplicates"] > 0:
        validation_results["warnings"].append(
            f"{validation_results['duplicates']['num_duplicates']} duplicati trovati"
        )
    
    # Verifica distribuzione classi
    if "is_balanced" in validation_results["class_distribution"]:
        if not validation_results["class_distribution"]["is_balanced"]:
            validation_results["warnings"].append(
                "Dataset non bilanciato - considerare tecniche di bilanciamento"
            )
    
    return validation_results


def generate_quality_report(
    df: pd.DataFrame,
    output_path: str,
    text_column: str = "text",
    label_column: str = "label",
) -> None:
    """
    Genera report completo qualità dati e lo salva.
    
    Args:
        df: DataFrame da analizzare
        output_path: Path dove salvare il report JSON
        text_column: Nome colonna testo
        label_column: Nome colonna label
    """
    validation = validate_dataset_quality(df, text_column, label_column)
    
    # Crea directory se non esiste
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Salva report JSON
    with open(output_path, "w") as f:
        json.dump(validation, f, indent=2, default=str)
    
    print(f"Report qualità dati salvato: {output_path}")
    
    # Stampa summary
    print("\n=== REPORT QUALITÀ DATI ===")
    print(f"Dataset size: {validation['dataset_size']}")
    print(f"\nDistribuzione classi:")
    if "class_counts" in validation["class_distribution"]:
        for cls, count in validation["class_distribution"]["class_counts"].items():
            pct = validation["class_distribution"]["class_percentages"].get(cls, 0)
            print(f"  {cls}: {count} ({pct:.2f}%)")
    
    print(f"\nStatistiche lunghezza testi:")
    for key, value in validation["text_length_stats"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    if validation["warnings"]:
        print(f"\n⚠️  Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
    
    if validation["errors"]:
        print(f"\n❌ Errors:")
        for error in validation["errors"]:
            print(f"  - {error}")

