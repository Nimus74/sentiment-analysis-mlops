"""
Utility per logging esperimenti su MLflow.
Gestisce il tracking di parametri, metriche, artifacts e modelli.
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import json


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "sentiment_analysis",
) -> None:
    """
    Configura MLflow con tracking URI e experiment.
    
    Args:
        tracking_uri: URI del tracking server (None = locale)
        experiment_name: Nome dell'esperimento
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Default: file system locale
        mlflow.set_tracking_uri("file:./mlruns")
    
    # Crea o recupera experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else None
    
    mlflow.set_experiment(experiment_name)


def log_config(config_path: str) -> None:
    """
    Logga il file di configurazione come artifact.
    
    Args:
        config_path: Path al file config.yaml
    """
    if os.path.exists(config_path):
        mlflow.log_artifact(config_path, "config")


def log_params(params: Dict[str, Any]) -> None:
    """
    Logga parametri del modello/esperimento.
    
    Args:
        params: Dizionario con parametri da loggare
    """
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Logga metriche di valutazione.
    
    Args:
        metrics: Dizionario con metriche da loggare
        step: Step/epoch corrente (opzionale)
    """
    if step is not None:
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    else:
        mlflow.log_metrics(metrics)


def log_model_artifact(
    model_path: str,
    artifact_path: str = "model",
    model_type: str = "pytorch",
    model_object=None,
) -> None:
    """
    Logga modello come artifact.
    
    Args:
        model_path: Path al modello salvato
        artifact_path: Path relativo nell'artifact store
        model_type: Tipo modello ('pytorch', 'sklearn', 'custom')
        model_object: Oggetto modello (opzionale, per PyTorch)
    """
    if model_type == "pytorch":
        if model_object is not None:
            # Se abbiamo l'oggetto modello, usalo direttamente
            mlflow.pytorch.log_model(model_object, artifact_path)
        else:
            # Altrimenti carica dal path
            import torch
            model = torch.load(model_path, map_location="cpu")
            mlflow.pytorch.log_model(model, artifact_path)
    elif model_type == "sklearn":
        mlflow.sklearn.log_model(model_path, artifact_path)
    else:
        # Custom: logga come file
        mlflow.log_artifacts(model_path, artifact_path)


def log_confusion_matrix(
    confusion_matrix: list,
    labels: list,
    artifact_path: str = "confusion_matrix",
) -> None:
    """
    Logga confusion matrix come JSON artifact.
    
    Args:
        confusion_matrix: Matrice di confusione (lista di liste)
        labels: Etichette delle classi
        artifact_path: Path relativo per l'artifact
    """
    cm_data = {
        "confusion_matrix": confusion_matrix,
        "labels": labels,
    }
    
    # Salva temporaneamente
    temp_path = Path("temp_cm.json")
    with open(temp_path, "w") as f:
        json.dump(cm_data, f, indent=2)
    
    mlflow.log_artifact(str(temp_path), artifact_path)
    temp_path.unlink()  # Rimuovi file temporaneo


def log_dataset_info(
    dataset_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    class_distribution: Dict[str, int],
) -> None:
    """
    Logga informazioni sul dataset utilizzato.
    
    Args:
        dataset_size: Dimensione totale dataset
        train_size: Dimensione training set
        val_size: Dimensione validation set
        test_size: Dimensione test set
        class_distribution: Distribuzione classi
    """
    mlflow.log_params({
        "dataset_size": dataset_size,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    })
    
    mlflow.log_params({
        f"class_distribution_{k}": v for k, v in class_distribution.items()
    })


def register_model(
    model_name: str,
    model_path: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
    stage: str = "Staging",
) -> str:
    """
    Registra modello nel Model Registry.
    
    Args:
        model_name: Nome del modello nel registry
        model_path: Path al modello (opzionale se già loggato)
        metrics: Metriche per descrizione
        stage: Stage iniziale ('Staging', 'Production', 'Archived')
    
    Returns:
        Model version URI
    """
    if model_path:
        model_uri = mlflow.register_model(model_path, model_name)
    else:
        # Usa modello loggato nella run corrente
        model_uri = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            model_name,
        )
    
    # Promuovi a stage se specificato
    if stage:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_uri.version,
            stage=stage,
        )
    
    return model_uri.model_uri


def get_best_model(
    experiment_name: str,
    metric: str = "macro_f1",
    ascending: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Recupera il miglior modello da un esperimento basato su una metrica.
    
    Args:
        experiment_name: Nome dell'esperimento
        metric: Nome della metrica per ordinamento
        ascending: True per minimizzare, False per massimizzare
    
    Returns:
        Dizionario con info del miglior modello o None
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return None
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )
    
    if runs.empty:
        return None
    
    best_run = runs.iloc[0]
    return {
        "run_id": best_run["run_id"],
        "metric_value": best_run[f"metrics.{metric}"],
        "model_uri": f"runs:/{best_run['run_id']}/model",
    }

