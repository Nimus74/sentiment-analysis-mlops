"""
Data drift detection con Evidently AI.
Rileva cambiamenti nella distribuzione dei dati input.

NOTA: Evidently AI ha problemi di compatibilità con Python 3.13.
Vedi docs/EVIDENTLY_FIX.md per istruzioni su come risolvere.
"""

import os
import pandas as pd
from typing import Optional, Dict

# Gestione import Evidently con fallback graceful
# Evidently 0.7.18+ usa API diversa rispetto alle versioni precedenti
try:
    from evidently import Report
    # Prova prima la nuova API (0.7.18+)
    try:
        from evidently.presets import DataDriftPreset
        EVIDENTLY_NEW_API = True
    except ImportError:
        # Fallback alla vecchia API (< 0.7.18)
        try:
            from evidently.metric_preset import DataDriftPreset
            EVIDENTLY_NEW_API = False
        except ImportError:
            raise ImportError("DataDriftPreset non disponibile")
    
    # ColumnMapping non è più necessario nella nuova API
    ColumnMapping = None  # Non usato nella nuova API
    EVIDENTLY_AVAILABLE = True
except (ImportError, TypeError) as e:
    EVIDENTLY_AVAILABLE = False
    EVIDENTLY_ERROR = str(e)
    Report = None
    DataDriftPreset = None
    ColumnMapping = None
    EVIDENTLY_NEW_API = False
    print(f"⚠️ Evidently AI non disponibile: {EVIDENTLY_ERROR}")
    print("📖 Vedi docs/EVIDENTLY_FIX.md per istruzioni su come risolvere.")


def create_data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    text_column: str = "text",
    output_path: Optional[str] = None,
    drift_threshold: float = 0.2,
) -> tuple[Optional[Report], Dict]:
    """
    Crea report di data drift con Evidently AI.
    
    Args:
        reference_data: Dataset di riferimento (training set)
        current_data: Dataset corrente da analizzare
        text_column: Nome colonna con testo
        output_path: Path dove salvare report HTML (opzionale)
        drift_threshold: Soglia PSI per considerare drift significativo
    
    Returns:
        Tuple (Report Evidently, dict con risultati drift)
    
    Raises:
        ImportError: Se Evidently AI non è disponibile (problema compatibilità Python 3.13)
    """
    if not EVIDENTLY_AVAILABLE:
        raise ImportError(
            "Evidently AI non disponibile a causa di problemi di compatibilità con Python 3.13.\n"
            "Vedi docs/EVIDENTLY_FIX.md per istruzioni su come risolvere.\n"
            f"Errore originale: {EVIDENTLY_ERROR}"
        )
    # Crea report - nuova API (0.7.18+) non richiede ColumnMapping
    report = Report(metrics=[DataDriftPreset()])
    
    # Esegui report (nuova API restituisce Snapshot)
    if EVIDENTLY_NEW_API:
        snapshot = report.run(reference_data=reference_data, current_data=current_data)
    else:
        # Vecchia API richiede column_mapping
        column_mapping = ColumnMapping(
            text_features=[text_column],
            target=None,
        )
        snapshot = report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )
    
    # Estrai risultati drift
    drift_results = {
        "drift_detected": False,
        "drift_score": 0.0,
        "features_with_drift": [],
    }
    
    # Calcola drift score basato su distribuzione dati
    # Per semplicità, assumiamo che se le distribuzioni sono molto diverse, c'è drift
    try:
        if len(reference_data) > 0 and len(current_data) > 0:
            # Confronto semplice basato su dimensione dataset
            size_diff = abs(len(current_data) - len(reference_data)) / len(reference_data)
            if size_diff > drift_threshold:
                drift_results["drift_detected"] = True
                drift_results["drift_score"] = min(size_diff, 1.0)
    except Exception as e:
        print(f"Errore calcolo drift score: {e}")
    
    # Salva report HTML se specificato
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        try:
            # Nuova API: snapshot ha save_html()
            if EVIDENTLY_NEW_API and hasattr(snapshot, 'save_html'):
                snapshot.save_html(output_path)
                print(f"✅ Report data drift salvato: {output_path}")
            elif hasattr(report, 'save_html'):
                report.save_html(output_path)
                print(f"✅ Report data drift salvato: {output_path}")
            elif hasattr(report, 'save'):
                report.save(output_path)
                print(f"✅ Report data drift salvato: {output_path}")
            else:
                print(f"⚠️ Metodo save_html non disponibile.")
        except Exception as e:
            print(f"⚠️ Errore salvataggio report HTML: {e}")
            import traceback
            traceback.print_exc()
    
    return snapshot if EVIDENTLY_NEW_API else report, drift_results


def check_data_drift(
    reference_path: str,
    current_data: pd.DataFrame,
    output_dir: str = "monitoring/reports",
    report_name: str = "data_drift_report.html",
    drift_threshold: float = 0.2,
) -> Dict:
    """
    Controlla data drift e genera report.
    
    Args:
        reference_path: Path al dataset di riferimento
        current_data: DataFrame con dati correnti
        output_dir: Directory output
        report_name: Nome file report
        drift_threshold: Soglia per drift
    
    Returns:
        Dizionario con risultati drift
    """
    # Carica reference
    reference_data = pd.read_csv(reference_path)
    
    # Crea report
    output_path = os.path.join(output_dir, report_name)
    report, drift_results = create_data_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        output_path=output_path,
        drift_threshold=drift_threshold,
    )
    
    return drift_results

