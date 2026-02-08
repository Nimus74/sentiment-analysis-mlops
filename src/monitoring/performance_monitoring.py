"""
Performance monitoring con Evidently AI.
Monitora performance modello in produzione se ground truth disponibile.
"""

import os
import pandas as pd
from typing import Optional, Dict

# Gestione import Evidently con fallback graceful
try:
    from evidently import Report, Dataset
    from evidently.presets import ClassificationPreset
    from evidently import DataDefinition, MulticlassClassification
    EVIDENTLY_AVAILABLE = True
    EVIDENTLY_NEW_API = True
except (ImportError, TypeError) as e:
    EVIDENTLY_AVAILABLE = False
    EVIDENTLY_NEW_API = False
    EVIDENTLY_ERROR = str(e)
    Report = None
    Dataset = None
    ClassificationPreset = None
    DataDefinition = None
    MulticlassClassification = None
    print(f"⚠️ Evidently AI non disponibile: {EVIDENTLY_ERROR}")


def create_performance_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_column: str = "label",
    prediction_column: str = "prediction",
    output_path: Optional[str] = None,
) -> tuple[Optional[Report], Dict]:
    """
    Crea report di performance modello.
    
    Args:
        reference_data: Dataset di riferimento (validation set con metriche)
        current_data: Dataset corrente con predizioni e ground truth
        target_column: Nome colonna con ground truth
        prediction_column: Nome colonna con predizioni
        output_path: Path dove salvare report HTML (opzionale)
    
    Returns:
        Tuple (Report Evidently, dict con metriche performance)
    """
    if not EVIDENTLY_AVAILABLE:
        raise ImportError(
            "Evidently AI non disponibile.\n"
            "Vedi docs/EVIDENTLY_FIX.md per istruzioni su come risolvere.\n"
            f"Errore originale: {EVIDENTLY_ERROR}"
        )
    
    if EVIDENTLY_NEW_API:
        # Nuova API (0.7.18+)
        data_def = DataDefinition(
            classification=[
                MulticlassClassification(
                    target=target_column,
                    prediction_labels=prediction_column,
                )
            ],
            categorical_columns=[target_column, prediction_column],
        )
        
        ref_ds = Dataset.from_pandas(reference_data, data_definition=data_def)
        cur_ds = Dataset.from_pandas(current_data, data_definition=data_def)
        
        report = Report([ClassificationPreset()])
        my_eval = report.run(cur_ds, ref_ds)
    else:
        # Vecchia API (< 0.7.18)
        from evidently import ColumnMapping
        
        column_mapping = ColumnMapping(
            target=target_column,
            prediction=prediction_column,
        )
        report = Report(metrics=[ClassificationPreset()])
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )
        my_eval = report
    
    # Estrai metriche performance
    performance_metrics = {
        "accuracy": None,
        "macro_f1": None,
        "precision": None,
        "recall": None,
        "degradation_detected": False,
    }
    
    # Calcola metriche manualmente da DataFrame
    try:
        from src.evaluation.metrics import calculate_metrics
        
        y_true = current_data[target_column].values
        y_pred = current_data[prediction_column].values
        
        metrics = calculate_metrics(y_true, y_pred)
        performance_metrics.update({
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "precision": metrics.get("macro_precision"),
            "recall": metrics.get("macro_recall"),
        })
    except Exception as e:
        print(f"Errore calcolo metriche: {e}")
    
    # Salva report HTML se specificato
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        try:
            if EVIDENTLY_NEW_API and hasattr(my_eval, 'save_html'):
                my_eval.save_html(output_path)
            elif hasattr(report, 'save_html'):
                report.save_html(output_path)
            print(f"Report performance salvato: {output_path}")
        except Exception as e:
            print(f"⚠️ Errore salvataggio report HTML: {e}")
    
    return my_eval if EVIDENTLY_NEW_API else report, performance_metrics


def monitor_performance(
    predictions_with_labels: pd.DataFrame,
    reference_path: str,
    output_dir: str = "monitoring/reports",
    report_name: str = "performance_report.html",
) -> Dict:
    """
    Monitora performance da predizioni con ground truth.
    
    Args:
        predictions_with_labels: DataFrame con predizioni e labels
        reference_path: Path al dataset di riferimento (validation)
        output_dir: Directory output
        report_name: Nome file report
    
    Returns:
        Dizionario con metriche performance
    """
    if not EVIDENTLY_AVAILABLE:
        print(f"⚠️ Impossibile monitorare performance: Evidently AI non disponibile.")
        return {
            "accuracy": None,
            "macro_f1": None,
            "error": EVIDENTLY_ERROR
        }
    
    # Carica reference
    reference_data = pd.read_csv(reference_path)
    
    # Crea report
    output_path = os.path.join(output_dir, report_name)
    report, performance_metrics = create_performance_report(
        reference_data=reference_data,
        current_data=predictions_with_labels,
        output_path=output_path,
    )
    
    return performance_metrics
