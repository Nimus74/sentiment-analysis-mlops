"""
Script per training/fine-tuning modello Transformer.
Supporta fine-tuning opzionale su dataset italiano.
"""

import argparse
import os
import yaml
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from src.models.transformer_model import TransformerSentimentModel
from src.training.mlflow_utils import (
    setup_mlflow,
    log_params,
    log_metrics,
    log_model_artifact,
    log_dataset_info,
)
from src.evaluation.metrics import calculate_metrics


class SentimentDataset(Dataset):
    """Dataset per sentiment analysis."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Calcola metriche per Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


def load_and_prepare_data(
    train_path: str,
    val_path: str,
    text_column: str = "text",
    label_column: str = "label",
):
    """Carica e prepara dati per training."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Trova colonna label se non specificata
    if label_column not in train_df.columns:
        for col in ["sentiment", "sentiment_label", "target"]:
            if col in train_df.columns:
                label_column = col
                break
    
    # Mapping label a numeri
    unique_labels = sorted(train_df[label_column].unique())
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    num_to_label = {i: label for label, i in label_to_num.items()}
    
    train_texts = train_df[text_column].tolist()
    train_labels = [label_to_num[l] for l in train_df[label_column]]
    
    val_texts = val_df[text_column].tolist()
    val_labels = [label_to_num[l] for l in val_df[label_column]]
    
    return (
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        label_to_num,
        num_to_label,
        train_df,
        val_df,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Transformer model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune model on Italian dataset",
    )
    args = parser.parse_args()
    
    # Carica configurazione
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    transformer_config = config.get("transformer", {})
    paths_config = config.get("paths", {})
    mlflow_config = config.get("mlflow", {})
    
    # Setup MLflow
    setup_mlflow(
        tracking_uri=mlflow_config.get("tracking_uri"),
        experiment_name=mlflow_config.get("experiment_name", "sentiment_analysis"),
    )
    
    # Paths
    data_processed = paths_config.get("data_processed", "data/processed")
    model_save_dir = paths_config.get("models_transformer", "models/transformer")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Se non si fa fine-tuning, usa modello pre-addestrato direttamente
    if not args.fine_tune:
        print("Usando modello pre-addestrato senza fine-tuning")
        print("Per fine-tuning, usa --fine-tune flag")
        
        # Valuta modello pre-addestrato su validation set
        model = TransformerSentimentModel(
            model_name=transformer_config.get(
                "model_name",
                "cardiffnlp/twitter-xlm-roberta-base-sentiment",  # Modello multilingue
            )
        )
        
        val_df = pd.read_csv(os.path.join(data_processed, "val.csv"))
        
        # Predizioni
        val_texts = val_df["text"].tolist()
        val_preds = model.predict_labels(val_texts)
        
        # Ground truth (converti label a numeri)
        unique_labels = sorted(val_df["label"].unique())
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        val_labels = np.array([label_to_num[l] for l in val_df["label"]])
        
        # Calcola metriche
        metrics = calculate_metrics(val_labels, val_preds, labels=unique_labels)
        
        import mlflow
        with mlflow.start_run(run_name="transformer_pretrained"):
            log_params(transformer_config)
            # Log solo metriche numeriche (escludi confusion_matrix)
            metrics_to_log = {k: v for k, v in metrics.items() if k != "confusion_matrix" and isinstance(v, (int, float))}
            log_metrics(metrics_to_log)
            log_dataset_info(
                dataset_size=len(val_df),
                train_size=0,
                val_size=len(val_df),
                test_size=0,
                class_distribution=val_df["label"].value_counts().to_dict(),
            )
        
        print("Metriche modello pre-addestrato:")
        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        return
    
    # Fine-tuning
    print("Inizio fine-tuning modello Transformer...")
    
    # Carica dati
    train_path = os.path.join(data_processed, "train.csv")
    val_path = os.path.join(data_processed, "val.csv")
    
    (
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        label_to_num,
        num_to_label,
        train_df,
        val_df,
    ) = load_and_prepare_data(train_path, val_path)
    
    # Carica tokenizer e modello
    model_name = transformer_config.get(
        "model_name",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",  # Modello multilingue
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_num),
    )
    
    # Crea dataset
    train_dataset = SentimentDataset(
        train_texts, train_labels, tokenizer, max_length=transformer_config.get("max_length", 128)
    )
    val_dataset = SentimentDataset(
        val_texts, val_labels, tokenizer, max_length=transformer_config.get("max_length", 128)
    )
    
    # Training arguments
    # Nota: evaluation_strategy è stato rinominato in eval_strategy nelle versioni recenti
    # Converti learning_rate a float se necessario
    lr = transformer_config.get("learning_rate", 2e-5)
    if isinstance(lr, str):
        # Se è una stringa come "2e-5", convertila
        lr = float(lr)
    
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        num_train_epochs=int(transformer_config.get("num_epochs", 3)),
        per_device_train_batch_size=int(transformer_config.get("batch_size", 16)),
        per_device_eval_batch_size=int(transformer_config.get("batch_size", 16)),
        learning_rate=lr,
        eval_strategy="epoch",  # Usa eval_strategy invece di evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=transformer_config.get("early_stopping_patience", 2),
                early_stopping_threshold=transformer_config.get("early_stopping_min_delta", 0.001),
            )
        ],
    )
    
    # Training
    import mlflow
    with mlflow.start_run(run_name="transformer_finetuned"):
        log_params(transformer_config)
        log_dataset_info(
            dataset_size=len(train_df) + len(val_df),
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=0,
            class_distribution=train_df["label"].value_counts().to_dict(),
        )
        
        trainer.train()
        
        # Valutazione finale
        eval_results = trainer.evaluate()
        log_metrics(eval_results)
        
        # Salva modello e tokenizer
        final_model_path = os.path.join(model_save_dir, "final_model")
        trainer.save_model(final_model_path)
        # Salva anche tokenizer
        tokenizer.save_pretrained(final_model_path)
        
        # Log modello su MLflow (usa il modello dal trainer)
        try:
            mlflow.pytorch.log_model(trainer.model, "model")
        except Exception as e:
            print(f"⚠️  Errore logging modello MLflow (continuo comunque): {e}")
            # Fallback: logga come artifact directory
            try:
                mlflow.log_artifacts(final_model_path, "model")
            except Exception as e2:
                print(f"⚠️  Errore anche nel fallback: {e2}")
        
        print(f"\n✅ Modello fine-tuned salvato: {final_model_path}")
        print(f"   Macro-F1: {eval_results['eval_macro_f1']:.4f}")
        print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")


if __name__ == "__main__":
    import mlflow
    from mlflow import start_run
    
    main()

