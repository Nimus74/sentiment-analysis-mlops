"""
Monitoring data quality con Evidently AI.
Genera report sulla qualità dei dati in input.

NOTA: Evidently AI ha problemi di compatibilità con Python 3.13.
Vedi docs/EVIDENTLY_FIX.md per istruzioni su come risolvere.
"""

import os
import pandas as pd
from typing import Optional

# Gestione import Evidently con fallback graceful
# Evidently 0.7.18+ usa API diversa rispetto alle versioni precedenti
try:
    from evidently import Report
    # Prova prima la nuova API (0.7.18+)
    try:
        from evidently.presets import DataSummaryPreset
        EVIDENTLY_NEW_API = True
        DataQualityPreset = DataSummaryPreset  # Alias per compatibilità
    except ImportError:
        # Fallback alla vecchia API (< 0.7.18)
        try:
            from evidently.metric_preset import DataQualityPreset
            EVIDENTLY_NEW_API = False
        except ImportError:
            raise ImportError("Nessun preset disponibile")
    
    # ColumnMapping non è più necessario nella nuova API
    ColumnMapping = None  # Non usato nella nuova API
    EVIDENTLY_AVAILABLE = True
except (ImportError, TypeError) as e:
    EVIDENTLY_AVAILABLE = False
    EVIDENTLY_ERROR = str(e)
    Report = None
    DataQualityPreset = None
    DataSummaryPreset = None
    ColumnMapping = None
    EVIDENTLY_NEW_API = False
    print(f"⚠️ Evidently AI non disponibile: {EVIDENTLY_ERROR}")
    print("📖 Vedi docs/EVIDENTLY_FIX.md per istruzioni su come risolvere.")


def create_data_quality_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    text_column: str = "text",
    output_path: Optional[str] = None,
) -> Optional[Report]:
    """
    Crea report di qualità dati con Evidently AI.
    
    Args:
        reference_data: Dataset di riferimento (training set)
        current_data: Dataset corrente da analizzare
        text_column: Nome colonna con testo
        output_path: Path dove salvare report HTML (opzionale)
    
    Returns:
        Report Evidently o None se Evidently non disponibile
    
    Raises:
        ImportError: Se Evidently AI non è disponibile (problema compatibilità Python 3.13)
    """
    if not EVIDENTLY_AVAILABLE:
        raise ImportError(
            "Evidently AI non disponibile.\n"
            "Vedi docs/EVIDENTLY_FIX.md per istruzioni su come risolvere.\n"
            f"Errore originale: {EVIDENTLY_ERROR}"
        )
    
    # Crea report - nuova API (0.7.18+) non richiede ColumnMapping
    report = Report(metrics=[DataQualityPreset()])
    
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
    
    # Salva report HTML se specificato
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        try:
            # Nuova API: snapshot ha save_html()
            if EVIDENTLY_NEW_API and hasattr(snapshot, 'save_html'):
                snapshot.save_html(output_path)
                print(f"✅ Report qualità dati salvato: {output_path}")
            elif hasattr(report, 'save_html'):
                report.save_html(output_path)
                print(f"✅ Report qualità dati salvato: {output_path}")
            elif hasattr(report, 'save'):
                report.save(output_path)
                print(f"✅ Report qualità dati salvato: {output_path}")
            else:
                print(f"⚠️ Metodo save_html non disponibile. Report creato ma non salvato.")
                print(f"   Usa Evidently UI server per visualizzare il report.")
        except Exception as e:
            print(f"⚠️ Errore salvataggio report HTML: {e}")
            import traceback
            traceback.print_exc()
            print(f"Report creato ma non salvato. Tipo snapshot: {type(snapshot)}")
    
    return snapshot if EVIDENTLY_NEW_API else report


def generate_data_quality_report(
    reference_path: str,
    current_data: pd.DataFrame,
    output_dir: str = "monitoring/reports",
    report_name: str = "data_quality_report.html",
) -> str:
    """
    Genera report qualità dati da file di riferimento.
    
    Args:
        reference_path: Path al dataset di riferimento (CSV)
        current_data: DataFrame con dati correnti
        output_dir: Directory output
        report_name: Nome file report
    
    Returns:
        Path al report generato
    """
    # Carica reference dataset
    reference_data = pd.read_csv(reference_path)
    
    # Crea report
    output_path = os.path.join(output_dir, report_name)
    report = create_data_quality_report(
        reference_data=reference_data,
        current_data=current_data,
        output_path=output_path,
    )
    
    return output_path

