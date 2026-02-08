"""
Script per retraining automatico modello FastText.
Trigger basato su data drift, performance degradation o schedule.
"""

import argparse
import os
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

from src.models.fasttext_model import FastTextSentimentModel
from src.data.preprocessing import prepare_fasttext_format
from src.training.mlflow_utils import (
    setup_mlflow,
    log_params,
    log_metrics,
    register_model,
    get_best_model,
)
from src.evaluation.metrics import calculate_metrics
import mlflow


def collect_new_data(
    feedback_file: str = "data/feedback.jsonl",
    api_logs_file: str = None,
    min_samples: int = 100,
) -> pd.DataFrame:
    """
    Raccoglie nuovi dati da feedback e log API.
    
    Args:
        feedback_file: Path al file feedback JSONL
        api_logs_file: Path ai log API (opzionale)
        min_samples: Numero minimo campioni per retraining
    
    Returns:
        DataFrame con nuovi dati
    """
    new_data = []
    
    # Carica feedback
    if os.path.exists(feedback_file):
        import json
        with open(feedback_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        feedback = json.loads(line)
                        if "actual_label" in feedback and feedback["actual_label"]:
                            new_data.append({
                                "text": feedback["text"],
                                "label": feedback["actual_label"],
                            })
                    except:
                        pass
    
    # Carica log API se disponibili
    if api_logs_file and os.path.exists(api_logs_file):
        # Implementa parsing log API se necessario
        pass
    
    if len(new_data) < min_samples:
        print(f"⚠️  Solo {len(new_data)} nuovi campioni disponibili (minimo: {min_samples})")
        return None
    
    return pd.DataFrame(new_data)


def retrain_fasttext(
    train_data_path: str,
    new_data: pd.DataFrame,
    val_data_path: str,
    output_path: str,
    config: dict,
) -> FastTextSentimentModel:
    """
    Retrain FastText combinando dati vecchi e nuovi.
    
    Args:
        train_data_path: Path al training set originale
        new_data: Nuovi dati da aggiungere
        val_data_path: Path al validation set
        output_path: Path dove salvare nuovo modello
        config: Configurazione FastText
    
    Returns:
        Modello retrained
    """
    # Carica training set originale
    train_df = pd.read_csv(train_data_path)
    
    # Combina con nuovi dati
    combined_df = pd.concat([train_df, new_data], ignore_index=True)
    
    # Prepara formato FastText
    temp_train_file = "data/processed/fasttext_retrain_temp.txt"
    prepare_fasttext_format(
        combined_df["text"].tolist(),
        combined_df["label"].tolist(),
        temp_train_file,
    )
    
    # Retrain
    model = FastTextSentimentModel.train(
        train_file=temp_train_file,
        output_path=output_path,
        lr=config.get("lr", 0.1),
        epoch=config.get("epoch", 25),
        wordNgrams=config.get("wordNgrams", 2),
        dim=config.get("dim", 100),
        minCount=config.get("minCount", 1),
        minn=config.get("minn", 3),
        maxn=config.get("maxn", 6),
        bucket=config.get("bucket", 2000000),
    )
    
    # Rimuovi file temporaneo
    if os.path.exists(temp_train_file):
        os.remove(temp_train_file)
    
    return model


def check_retraining_triggers(
    config: dict,
    data_drift_detected: bool = False,
    performance_degraded: bool = False,
    last_retrain_date: datetime = None,
) -> bool:
    """
    Verifica se ci sono trigger per retraining.
    
    Args:
        config: Configurazione retraining
        data_drift_detected: Se data drift è stato rilevato
        performance_degraded: Se performance è degradata
        last_retrain_date: Data ultimo retraining
    
    Returns:
        True se retraining necessario
    """
    triggers_config = config.get("retraining", {}).get("triggers", {})
    
    # Trigger data drift
    if triggers_config.get("data_drift", False) and data_drift_detected:
        return True
    
    # Trigger performance degradation
    if triggers_config.get("performance_degradation", False) and performance_degraded:
        return True
    
    # Trigger schedulato
    if triggers_config.get("scheduled", False):
        if last_retrain_date is None:
            return True  # Mai fatto retraining
        
        interval_days = triggers_config.get("schedule_interval_days", 30)
        days_since_retrain = (datetime.now() - last_retrain_date).days
        
        if days_since_retrain >= interval_days:
            return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Retrain FastText model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forza retraining anche senza trigger",
    )
    args = parser.parse_args()
    
    # Carica configurazione
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    paths_config = config.get("paths", {})
    mlflow_config = config.get("mlflow", {})
    retraining_config = config.get("retraining", {})
    
    # Setup MLflow
    setup_mlflow(
        tracking_uri=mlflow_config.get("tracking_uri"),
        experiment_name=mlflow_config.get("experiment_name", "sentiment_analysis"),
    )
    
    # Paths
    data_processed = paths_config.get("data_processed", "data/processed")
    model_save_dir = paths_config.get("models_fasttext", "models/fasttext")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Verifica trigger (se non forzato)
    if not args.force:
        # Qui potresti controllare data drift, performance, etc.
        # Per semplicità, assumiamo che i trigger siano già stati verificati
        # da altri script di monitoring
        pass
    
    # Raccogli nuovi dati
    print("Raccolta nuovi dati...")
    new_data = collect_new_data(
        min_samples=retraining_config.get("fasttext", {}).get("min_new_samples", 100)
    )
    
    if new_data is None:
        print("❌ Dati insufficienti per retraining")
        return
    
    print(f"✅ Raccolti {len(new_data)} nuovi campioni")
    
    # Paths dati
    train_path = os.path.join(data_processed, "train.csv")
    val_path = os.path.join(data_processed, "val.csv")
    
    # Retrain
    print("\nInizio retraining...")
    new_model_path = os.path.join(model_save_dir, f"fasttext_model_retrained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin")
    
    model = retrain_fasttext(
        train_data_path=train_path,
        new_data=new_data,
        val_data_path=val_path,
        output_path=new_model_path,
        config=config.get("fasttext", {}),
    )
    
    # Valuta nuovo modello
    print("\nValutazione nuovo modello...")
    val_df = pd.read_csv(val_path)
    val_texts = val_df["text"].tolist()
    val_preds = model.predict_labels(val_texts)
    
    unique_labels = sorted(val_df["label"].unique())
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    val_labels = pd.Series([label_to_num[l] for l in val_df["label"]]).values
    
    new_metrics = calculate_metrics(val_labels, val_preds, labels=unique_labels)
    
    # Confronta con modello corrente
    current_model_path = os.path.join(model_save_dir, "fasttext_model.bin")
    if os.path.exists(current_model_path):
        current_model = FastTextSentimentModel.load(current_model_path)
        current_preds = current_model.predict_labels(val_texts)
        current_metrics = calculate_metrics(val_labels, current_preds, labels=unique_labels)
        
        improvement = new_metrics["macro_f1"] - current_metrics["macro_f1"]
        print(f"\nConfronto modelli:")
        print(f"  Corrente Macro-F1: {current_metrics['macro_f1']:.4f}")
        print(f"  Nuovo Macro-F1: {new_metrics['macro_f1']:.4f}")
        print(f"  Miglioramento: {improvement:+.4f}")
        
        # Verifica criteri promozione
        promotion_config = retraining_config.get("promotion", {})
        min_improvement = promotion_config.get("min_improvement_f1", 0.02)
        
        if improvement >= min_improvement:
            print(f"\n✅ Nuovo modello migliore! Promozione...")
            # Sostituisci modello corrente
            import shutil
            shutil.copy(new_model_path, current_model_path)
            print(f"   Modello promosso: {current_model_path}")
        else:
            print(f"\n⚠️  Miglioramento insufficiente (< {min_improvement})")
            print(f"   Nuovo modello salvato ma non promosso: {new_model_path}")
    else:
        print("\n✅ Primo modello, nessun confronto possibile")
        import shutil
        shutil.copy(new_model_path, current_model_path)
    
    # Log su MLflow
    with mlflow.start_run(run_name=f"fasttext_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        log_params(config.get("fasttext", {}))
        log_metrics(new_metrics)
        mlflow.log_artifact(new_model_path, "model")
    
    print("\n✅ Retraining completato!")


if __name__ == "__main__":
    main()

