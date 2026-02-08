"""
Script per confronto comparativo tra Transformer e FastText.
Valuta entrambi i modelli sullo stesso test set con stesse metriche.
"""

import argparse
import os
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import mlflow

from src.models.transformer_model import TransformerSentimentModel
from src.models.fasttext_model import FastTextSentimentModel
from src.evaluation.metrics import (
    calculate_metrics,
    compare_models_metrics,
    get_classification_report,
)
from src.training.mlflow_utils import (
    setup_mlflow,
    log_metrics,
    log_confusion_matrix,
)


def load_models(
    transformer_path: str = None,
    fasttext_path: str = None,
    transformer_model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",  # Modello multilingue
):
    """Carica entrambi i modelli."""
    print("Caricamento modelli...")
    
    # Transformer
    if transformer_path and os.path.exists(transformer_path):
        transformer = TransformerSentimentModel.load(transformer_path)
    else:
        transformer = TransformerSentimentModel(model_name=transformer_model_name)
    
    # FastText
    if fasttext_path and os.path.exists(fasttext_path):
        fasttext_model = FastTextSentimentModel.load(fasttext_path)
    else:
        raise ValueError(f"Modello FastText non trovato: {fasttext_path}")
    
    print("✅ Modelli caricati")
    return transformer, fasttext_model


def evaluate_model(
    model,
    texts: list,
    labels: list,
    model_name: str,
    label_to_num: dict,
):
    """Valuta un modello e ritorna metriche."""
    print(f"\nValutazione {model_name}...")
    
    # Predizioni
    if hasattr(model, "predict_labels"):
        preds = model.predict_labels(texts)
    else:
        # Fallback per modelli senza predict_labels
        predictions = model.predict_batch(texts)
        label_to_num_pred = {"negative": 0, "neutral": 1, "positive": 2}
        preds = np.array([label_to_num_pred[p["label"]] for p in predictions])
    
    # Converti labels a numeri
    labels_num = np.array([label_to_num[l] for l in labels])
    
    # Calcola metriche
    unique_labels = sorted(label_to_num.keys())
    metrics = calculate_metrics(labels_num, preds, labels=unique_labels)
    
    # Classification report
    report = get_classification_report(labels_num, preds, labels=unique_labels)
    
    return metrics, preds, report


def plot_confusion_matrices(
    cm_transformer: np.ndarray,
    cm_fasttext: np.ndarray,
    labels: list,
    output_path: str,
):
    """Plotta confusion matrix per entrambi i modelli."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Transformer
    sns.heatmap(
        cm_transformer,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("Transformer Confusion Matrix")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    
    # FastText
    sns.heatmap(
        cm_fasttext,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title("FastText Confusion Matrix")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrices salvate: {output_path}")
    plt.close()


def statistical_significance_test(
    labels_true: np.ndarray,
    preds_transformer: np.ndarray,
    preds_fasttext: np.ndarray,
) -> dict:
    """
    Test di significatività statistica tra i due modelli.
    Usa McNemar test per test binari accoppiati.
    """
    # Per multiclasse, calcola accuracy per ogni classe
    results = {}
    
    for class_label in [0, 1, 2]:
        # Crea problemi binari: classe vs altre
        y_true_binary = (labels_true == class_label).astype(int)
        y_pred_transformer_binary = (preds_transformer == class_label).astype(int)
        y_pred_fasttext_binary = (preds_fasttext == class_label).astype(int)
        
        # Conta accordi/disaccordi
        both_correct = ((y_true_binary == y_pred_transformer_binary) &
                        (y_true_binary == y_pred_fasttext_binary)).sum()
        transformer_correct = ((y_true_binary == y_pred_transformer_binary) &
                                (y_true_binary != y_pred_fasttext_binary)).sum()
        fasttext_correct = ((y_true_binary != y_pred_transformer_binary) &
                             (y_true_binary == y_pred_fasttext_binary)).sum()
        both_wrong = ((y_true_binary != y_pred_transformer_binary) &
                       (y_true_binary != y_pred_fasttext_binary)).sum()
        
        # McNemar test
        contingency_table = [[both_correct, transformer_correct],
                             [fasttext_correct, both_wrong]]
        
        try:
            chi2, p_value = stats.mcnemar(contingency_table, exact=False, correction=True)
            results[f"class_{class_label}"] = {
                "chi2": float(chi2),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }
        except:
            results[f"class_{class_label}"] = {
                "chi2": None,
                "p_value": None,
                "significant": None,
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Transformer vs FastText")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--transformer-path",
        type=str,
        default=None,
        help="Path to fine-tuned Transformer model (optional)",
    )
    parser.add_argument(
        "--fasttext-path",
        type=str,
        default="models/fasttext/fasttext_model.bin",
        help="Path to FastText model",
    )
    args = parser.parse_args()
    
    # Carica configurazione
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    paths_config = config.get("paths", {})
    mlflow_config = config.get("mlflow", {})
    
    # Setup MLflow
    setup_mlflow(
        tracking_uri=mlflow_config.get("tracking_uri"),
        experiment_name=mlflow_config.get("experiment_name", "sentiment_analysis"),
    )
    
    # Paths
    data_processed = paths_config.get("data_processed", "data/processed")
    reports_dir = paths_config.get("reports", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Carica test set
    test_path = os.path.join(data_processed, "test.csv")
    test_df = pd.read_csv(test_path)
    
    print(f"Test set: {len(test_df)} campioni")
    
    # Mapping label
    unique_labels = sorted(test_df["label"].unique())
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()
    
    # Carica modelli
    transformer_model_name = config.get("transformer", {}).get(
        "model_name",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",  # Modello multilingue
    )
    
    transformer, fasttext_model = load_models(
        transformer_path=args.transformer_path,
        fasttext_path=args.fasttext_path,
        transformer_model_name=transformer_model_name,
    )
    
    # Valuta entrambi i modelli
    metrics_transformer, preds_transformer, report_transformer = evaluate_model(
        transformer, test_texts, test_labels, "Transformer", label_to_num
    )
    
    metrics_fasttext, preds_fasttext, report_fasttext = evaluate_model(
        fasttext_model, test_texts, test_labels, "FastText", label_to_num
    )
    
    # Confronto metriche
    comparison_df = compare_models_metrics(
        metrics_transformer,
        metrics_fasttext,
        model1_name="Transformer",
        model2_name="FastText",
    )
    
    print("\n" + "=" * 60)
    print("CONFRONTO MODELLI")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    
    # Test significatività statistica
    labels_num = np.array([label_to_num[l] for l in test_labels])
    significance_results = statistical_significance_test(
        labels_num, preds_transformer, preds_fasttext
    )
    
    print("\nTest di Significatività Statistica (McNemar):")
    for class_name, result in significance_results.items():
        if result["p_value"] is not None:
            sig = "✓" if result["significant"] else "✗"
            print(
                f"  {class_name}: p-value={result['p_value']:.4f} {sig}"
            )
    
    # Confusion matrices
    cm_transformer = np.array(metrics_transformer["confusion_matrix"])
    cm_fasttext = np.array(metrics_fasttext["confusion_matrix"])
    
    cm_path = os.path.join(reports_dir, "model_comparison", "confusion_matrices.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plot_confusion_matrices(cm_transformer, cm_fasttext, unique_labels, cm_path)
    
    # Salva report completo
    report_path = os.path.join(reports_dir, "model_comparison", "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("REPORT CONFRONTO MODELLI\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METRICHE COMPARATIVE\n")
        f.write("-" * 60 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("TRANSFORMER - Classification Report\n")
        f.write("-" * 60 + "\n")
        f.write(report_transformer)
        f.write("\n\n")
        
        f.write("FASTTEXT - Classification Report\n")
        f.write("-" * 60 + "\n")
        f.write(report_fasttext)
        f.write("\n\n")
        
        f.write("TEST SIGNIFICATIVITÀ STATISTICA\n")
        f.write("-" * 60 + "\n")
        for class_name, result in significance_results.items():
            f.write(f"{class_name}: {result}\n")
    
    print(f"\n✅ Report completo salvato: {report_path}")
    
    # Log su MLflow
    try:
        with mlflow.start_run(run_name="model_comparison"):
            mlflow.log_metrics({
                "transformer_macro_f1": metrics_transformer["macro_f1"],
                "fasttext_macro_f1": metrics_fasttext["macro_f1"],
                "transformer_accuracy": metrics_transformer["accuracy"],
                "fasttext_accuracy": metrics_fasttext["accuracy"],
            })
            
            log_confusion_matrix(
                metrics_transformer["confusion_matrix"],
                unique_labels,
                "transformer_confusion_matrix",
            )
            
            log_confusion_matrix(
                metrics_fasttext["confusion_matrix"],
                unique_labels,
                "fasttext_confusion_matrix",
            )
            
            mlflow.log_artifact(report_path, "comparison_report")
            mlflow.log_artifact(cm_path, "confusion_matrices")
    except Exception as e:
        print(f"\n⚠️  Errore logging MLflow (continuo comunque): {e}")


if __name__ == "__main__":
    main()

