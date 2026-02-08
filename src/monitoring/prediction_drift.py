"""
Prediction drift monitoring con Evidently AI.
Monitora distribuzione delle predizioni nel tempo.

NOTA: Evidently AI ha problemi di compatibilità con Python 3.13.
Vedi docs/EVIDENTLY_FIX.md per istruzioni su come risolvere.
"""

import os
import pandas as pd
from typing import Optional, Dict, List

# Gestione import Evidently con fallback graceful
# Evidently 0.7.18+ usa API diversa rispetto alle versioni precedenti
try:
    from evidently import Report, Dataset
    # Prova prima la nuova API (0.7.18+)
    try:
        from evidently.presets import ClassificationPreset
        from evidently import DataDefinition, MulticlassClassification
        ClassificationPerformancePreset = ClassificationPreset  # Alias per compatibilità
        EVIDENTLY_NEW_API = True
    except ImportError:
        # Fallback alla vecchia API (< 0.7.18)
        try:
            from evidently.metric_preset import ClassificationPerformancePreset
            EVIDENTLY_NEW_API = False
            Dataset = None
            DataDefinition = None
            MulticlassClassification = None
        except ImportError:
            raise ImportError("ClassificationPerformancePreset non disponibile")
    
    # ColumnMapping non è più necessario nella nuova API
    ColumnMapping = None  # Non usato nella nuova API
    EVIDENTLY_AVAILABLE = True
except (ImportError, TypeError) as e:
    EVIDENTLY_AVAILABLE = False
    EVIDENTLY_ERROR = str(e)
    Report = None
    Dataset = None
    ClassificationPerformancePreset = None
    ColumnMapping = None
    DataDefinition = None
    MulticlassClassification = None
    EVIDENTLY_NEW_API = False
    print(f"⚠️ Evidently AI non disponibile: {EVIDENTLY_ERROR}")
    print("📖 Vedi docs/EVIDENTLY_FIX.md per istruzioni su come risolvere.")


def create_prediction_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    prediction_column: str = "prediction",
    target_column: Optional[str] = None,
    output_path: Optional[str] = None,
) -> tuple[Optional[Report], Dict]:
    """
    Crea report di prediction drift.
    
    Args:
        reference_data: Dataset di riferimento con predizioni
        current_data: Dataset corrente con predizioni
        prediction_column: Nome colonna con predizioni
        target_column: Nome colonna con target/ground truth (opzionale, ma richiesto per ClassificationPreset)
        output_path: Path dove salvare report HTML (opzionale)
    
    Returns:
        Tuple (Report Evidently, dict con risultati drift)
    
    Raises:
        ImportError: Se Evidently AI non è disponibile
        ValueError: Se le colonne richieste non sono presenti nel dataset
    """
    if not EVIDENTLY_AVAILABLE:
        raise ImportError(
            "Evidently AI non disponibile.\n"
            "Vedi docs/EVIDENTLY_FIX.md per istruzioni su come risolvere.\n"
            f"Errore originale: {EVIDENTLY_ERROR}"
        )
    
    # Verifica che le colonne necessarie esistano
    if prediction_column not in reference_data.columns:
        raise ValueError(f"Colonna '{prediction_column}' non trovata nel dataset di riferimento")
    if prediction_column not in current_data.columns:
        raise ValueError(f"Colonna '{prediction_column}' non trovata nel dataset corrente")
    
    # Nuova API (0.7.18+) richiede DataDefinition e Dataset
    if EVIDENTLY_NEW_API:
        # Se target_column non specificato, prova a trovarlo automaticamente
        if target_column is None:
            for col in ["label", "target", "y_true", "ground_truth"]:
                if col in reference_data.columns:
                    target_column = col
                    break
        
        # Se abbiamo sia target che prediction, usa ClassificationPreset
        if target_column and target_column in reference_data.columns:
            # Crea DataDefinition per classificazione multiclasse
            data_def = DataDefinition(
                classification=[
                    MulticlassClassification(
                        target=target_column,
                        prediction_labels=prediction_column,
                    )
                ],
                categorical_columns=[prediction_column, target_column],
            )
            
            # Crea Dataset con DataDefinition
            ref_ds = Dataset.from_pandas(reference_data, data_definition=data_def)
            cur_ds = Dataset.from_pandas(current_data, data_definition=data_def)
            
            # Crea report con ClassificationPreset
            report = Report([ClassificationPerformancePreset()])
            my_eval = report.run(cur_ds, ref_ds)
            
        else:
            # Se non c'è target, usa DataDriftPreset per monitorare solo la distribuzione delle predizioni
            from evidently.presets import DataDriftPreset
            
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=reference_data[[prediction_column]], 
                current_data=current_data[[prediction_column]]
            )
            my_eval = report
            
    else:
        # Vecchia API (< 0.7.18)
        from evidently import ColumnMapping
        
        column_mapping = ColumnMapping(
            prediction=prediction_column,
            target=target_column,
        )
        report = Report(metrics=[ClassificationPerformancePreset()])
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )
        my_eval = report
    
    # Analizza distribuzione predizioni
    drift_results = {
        "drift_detected": False,
        "drift_score": 0.0,
        "reference_distribution": {},
        "current_distribution": {},
        "distribution_shift": {},
    }
    
    # Calcola distribuzione classi
    try:
        ref_dist = reference_data[prediction_column].value_counts(normalize=True).to_dict()
        curr_dist = current_data[prediction_column].value_counts(normalize=True).to_dict()
        
        drift_results["reference_distribution"] = ref_dist
        drift_results["current_distribution"] = curr_dist
        
        # Calcola shift nella distribuzione
        all_labels = set(ref_dist.keys()) | set(curr_dist.keys())
        max_shift = 0.0
        
        for label in all_labels:
            ref_pct = ref_dist.get(label, 0.0)
            curr_pct = curr_dist.get(label, 0.0)
            shift = abs(curr_pct - ref_pct)
            drift_results["distribution_shift"][label] = float(shift)
            max_shift = max(max_shift, shift)
            
            # Considera drift se shift > 15%
            if shift > 0.15:
                drift_results["drift_detected"] = True
        
        drift_results["drift_score"] = float(max_shift)
        
    except Exception as e:
        print(f"Errore calcolo distribuzione: {e}")
    
    # Salva report HTML se specificato
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        try:
            if EVIDENTLY_NEW_API and hasattr(my_eval, 'save_html'):
                my_eval.save_html(output_path)
                print(f"✅ Report prediction drift salvato: {output_path}")
            elif hasattr(report, 'save_html'):
                report.save_html(output_path)
                print(f"✅ Report prediction drift salvato: {output_path}")
            else:
                print(f"⚠️ Salvataggio HTML non disponibile nella nuova API.")
                print(f"   Usa Evidently UI server per visualizzare il report.")
        except Exception as e:
            print(f"⚠️ Errore salvataggio report HTML: {e}")
    
    return my_eval if EVIDENTLY_NEW_API else report, drift_results


def check_prediction_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_dir: str = "monitoring/reports",
    report_name: str = "prediction_drift_report.html",
) -> Dict:
    """
    Controlla prediction drift e genera report.
    
    Args:
        reference_data: DataFrame con dati di riferimento (con predizioni)
        current_data: DataFrame con dati correnti (con predizioni)
        output_dir: Directory output
        report_name: Nome file report
    
    Returns:
        Dizionario con risultati drift
    """
    if not EVIDENTLY_AVAILABLE:
        print(f"⚠️ Impossibile controllare prediction drift: Evidently AI non disponibile.")
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "reference_distribution": {},
            "current_distribution": {},
            "distribution_shift": {},
            "error": EVIDENTLY_ERROR
        }

    # Crea report
    output_path = os.path.join(output_dir, report_name)
    report, drift_results = create_prediction_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        output_path=output_path,
    )
    
    return drift_results


def monitor_predictions(
    predictions_log: List[Dict],
    reference_distribution: Dict[str, float],
    output_dir: str = "monitoring/reports",
    report_name: str = "prediction_drift_report.html",
) -> Dict:
    """
    Monitora predizioni da log API.
    
    Args:
        predictions_log: Lista di dizionari con predizioni (es. da API logs)
        reference_distribution: Distribuzione di riferimento
        output_dir: Directory output
        report_name: Nome file report
    
    Returns:
        Dizionario con risultati drift
    """
    if not EVIDENTLY_AVAILABLE:
        print(f"⚠️ Impossibile monitorare predizioni: Evidently AI non disponibile.")
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "error": EVIDENTLY_ERROR
        }
    
    # Converti log in DataFrame
    current_df = pd.DataFrame(predictions_log)
    
    # Verifica che ci sia una colonna 'prediction' o 'label'
    prediction_col = None
    for col in ["prediction", "label", "predicted_label"]:
        if col in current_df.columns:
            prediction_col = col
            break
    
    if prediction_col is None:
        raise ValueError("Nessuna colonna di predizione trovata nel log")
    
    # Crea reference DataFrame dalla distribuzione
    # Genera campioni basati sulla distribuzione di riferimento
    total_samples = len(current_df)
    reference_samples = []
    for label, pct in reference_distribution.items():
        count = int(total_samples * pct)
        reference_samples.extend([label] * count)
    
    # Aggiungi campioni se necessario per raggiungere total_samples
    remaining = total_samples - len(reference_samples)
    if remaining > 0:
        # Distribuisci i rimanenti proporzionalmente
        for label, pct in reference_distribution.items():
            if remaining <= 0:
                break
            reference_samples.append(label)
            remaining -= 1
    
    reference_df = pd.DataFrame({
        prediction_col: reference_samples[:total_samples]
    })
    
    # Crea report
    output_path = os.path.join(output_dir, report_name)
    report, drift_results = create_prediction_drift_report(
        reference_data=reference_df,
        current_data=current_df,
        prediction_column=prediction_col,
        output_path=output_path,
    )
    
    return drift_results
