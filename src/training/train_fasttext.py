"""
Script per training modello FastText supervised.
"""

import argparse
import os
import yaml
from pathlib import Path
import pandas as pd

from src.models.fasttext_model import FastTextSentimentModel
from src.data.preprocessing import prepare_fasttext_format
from src.training.mlflow_utils import (
    setup_mlflow,
    log_params,
    log_metrics,
    log_model_artifact,
    log_dataset_info,
)
from src.evaluation.metrics import calculate_metrics
import mlflow


def prepare_fasttext_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    output_dir: str = "data/processed",
) -> tuple:
    """
    Prepara file nel formato FastText per training e validation.
    
    Returns:
        Tuple (train_file_path, val_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, "fasttext_train.txt")
    val_file = os.path.join(output_dir, "fasttext_val.txt")
    
    # Prepara formato FastText
    train_texts = train_df[text_column].tolist()
    train_labels = train_df[label_column].tolist()
    prepare_fasttext_format(train_texts, train_labels, train_file)
    
    val_texts = val_df[text_column].tolist()
    val_labels = val_df[label_column].tolist()
    prepare_fasttext_format(val_texts, val_labels, val_file)
    
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(description="Train FastText model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    
    # Carica configurazione
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    fasttext_config = config.get("fasttext", {})
    paths_config = config.get("paths", {})
    mlflow_config = config.get("mlflow", {})
    
    # Setup MLflow
    setup_mlflow(
        tracking_uri=mlflow_config.get("tracking_uri"),
        experiment_name=mlflow_config.get("experiment_name", "sentiment_analysis"),
    )
    
    # Paths
    data_processed = paths_config.get("data_processed", "data/processed")
    model_save_dir = paths_config.get("models_fasttext", "models/fasttext")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Carica dati
    train_path = os.path.join(data_processed, "train.csv")
    val_path = os.path.join(data_processed, "val.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Prepara file FastText
    train_file, val_file = prepare_fasttext_data(
        train_df, val_df, output_dir=data_processed
    )
    
    # Path modello output
    model_output_path = os.path.join(model_save_dir, "fasttext_model.bin")
    
    # Training
    print("\nInizio training FastText...")
    
    with mlflow.start_run(run_name="fasttext_supervised"):
        # Log parametri
        log_params(fasttext_config)
        log_dataset_info(
            dataset_size=len(train_df) + len(val_df),
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=0,
            class_distribution=train_df["label"].value_counts().to_dict(),
        )
        
        # Addestra modello
        model = FastTextSentimentModel.train(
            train_file=train_file,
            output_path=model_output_path,
            lr=fasttext_config.get("lr", 0.1),
            epoch=fasttext_config.get("epoch", 25),
            wordNgrams=fasttext_config.get("wordNgrams", 2),
            dim=fasttext_config.get("dim", 100),
            minCount=fasttext_config.get("minCount", 1),
            minn=fasttext_config.get("minn", 3),
            maxn=fasttext_config.get("maxn", 6),
            bucket=fasttext_config.get("bucket", 2000000),
        )
        
        # Valutazione su validation set
        print("\nValutazione su validation set...")
        val_texts = val_df["text"].tolist()
        val_preds = model.predict_labels(val_texts)
        
        # Ground truth
        unique_labels = sorted(val_df["label"].unique())
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        val_labels = [label_to_num[l] for l in val_df["label"]]
        val_labels = pd.Series(val_labels).values
        
        # Calcola metriche
        metrics = calculate_metrics(val_labels, val_preds, labels=unique_labels)
        
        # Log metriche (escludi confusion_matrix che è una lista)
        metrics_to_log = {k: v for k, v in metrics.items() if k != "confusion_matrix" and isinstance(v, (int, float))}
        log_metrics(metrics_to_log)
        
        # Log modello
        log_model_artifact(model_output_path, "model", "custom")
        
        print("\n✅ Training completato!")
        print(f"   Modello salvato: {model_output_path}")
        print(f"   Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nMetriche per classe:")
        for label in unique_labels:
            print(
                f"   {label}: F1={metrics.get(f'{label}_f1', 0):.4f}, "
                f"Precision={metrics.get(f'{label}_precision', 0):.4f}, "
                f"Recall={metrics.get(f'{label}_recall', 0):.4f}"
            )


if __name__ == "__main__":
    main()

