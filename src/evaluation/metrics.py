"""
Modulo per calcolo metriche di valutazione.
Implementa le metriche principali e secondarie per il confronto modelli.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import pandas as pd


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Calcola tutte le metriche di valutazione.
    
    Args:
        y_true: Etichette vere (ground truth)
        y_pred: Predizioni del modello
        labels: Lista delle classi (es. ['negative', 'neutral', 'positive'])
        average: Tipo di average per F1 ('macro', 'micro', 'weighted')
    
    Returns:
        Dizionario con tutte le metriche calcolate
    """
    if labels is None:
        labels = ["negative", "neutral", "positive"]
    
    # Metriche base
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 per classe e macro/micro
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0
    )
    
    # Macro F1 (metrica principale)
    macro_f1 = np.mean(f1)
    
    # Micro F1
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), average="micro", zero_division=0
    )
    
    # Weighted F1
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), average="weighted", zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    # Metriche per classe
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[f"{label}_precision"] = float(precision[i])
        per_class_metrics[f"{label}_recall"] = float(recall[i])
        per_class_metrics[f"{label}_f1"] = float(f1[i])
        per_class_metrics[f"{label}_support"] = int(support[i])
    
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "weighted_f1": float(weighted_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "confusion_matrix": cm.tolist(),
        **per_class_metrics,
    }
    
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> str:
    """
    Genera report di classificazione testuale.
    
    Args:
        y_true: Etichette vere
        y_pred: Predizioni
        labels: Nomi delle classi
    
    Returns:
        Report testuale formattato
    """
    if labels is None:
        labels = ["negative", "neutral", "positive"]
    
    target_names = labels
    return classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )


def compare_models_metrics(
    metrics_model1: Dict[str, float],
    metrics_model2: Dict[str, float],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
) -> pd.DataFrame:
    """
    Confronta metriche di due modelli in una tabella.
    
    Args:
        metrics_model1: Metriche del primo modello
        metrics_model2: Metriche del secondo modello
        model1_name: Nome del primo modello
        model2_name: Nome del secondo modello
    
    Returns:
        DataFrame con confronto metriche
    """
    comparison = {
        "Metric": [],
        model1_name: [],
        model2_name: [],
        "Difference": [],
    }
    
    # Metriche da confrontare
    metrics_to_compare = [
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "micro_f1",
        "weighted_f1",
    ]
    
    for metric in metrics_to_compare:
        if metric in metrics_model1 and metric in metrics_model2:
            val1 = metrics_model1[metric]
            val2 = metrics_model2[metric]
            diff = val2 - val1
            
            comparison["Metric"].append(metric)
            comparison[model1_name].append(f"{val1:.4f}")
            comparison[model2_name].append(f"{val2:.4f}")
            comparison["Difference"].append(f"{diff:+.4f}")
    
    return pd.DataFrame(comparison)


def check_metrics_thresholds(
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
) -> Tuple[bool, List[str]]:
    """
    Verifica se le metriche soddisfano le soglie richieste.
    
    Args:
        metrics: Metriche calcolate
        thresholds: Soglie da verificare (es. {"macro_f1_min": 0.75})
    
    Returns:
        Tuple (passa_tutti_i_test, lista_messaggi)
    """
    passes = True
    messages = []
    
    # Verifica macro-F1 minimo
    if "macro_f1_min" in thresholds:
        macro_f1 = metrics.get("macro_f1", 0.0)
        threshold = thresholds["macro_f1_min"]
        if macro_f1 < threshold:
            passes = False
            messages.append(
                f"Macro-F1 ({macro_f1:.4f}) < threshold ({threshold:.4f})"
            )
        else:
            messages.append(
                f"✓ Macro-F1 ({macro_f1:.4f}) >= threshold ({threshold:.4f})"
            )
    
    # Verifica F1 per classe minimo
    if "per_class_f1_min" in thresholds:
        threshold = thresholds["per_class_f1_min"]
        for key, value in metrics.items():
            if key.endswith("_f1") and not key.startswith("macro") and not key.startswith("micro"):
                if value < threshold:
                    passes = False
                    messages.append(
                        f"{key} ({value:.4f}) < threshold ({threshold:.4f})"
                    )
    
    return passes, messages

