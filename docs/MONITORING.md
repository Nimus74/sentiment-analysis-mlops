# Guida Monitoring con Evidently AI

## Overview

Il sistema usa Evidently AI per monitoring continuo di:
- **Data Quality**: Qualità dati in input
- **Data Drift**: Cambiamenti distribuzione dati
- **Prediction Drift**: Cambiamenti distribuzione predizioni
- **Performance**: Metriche modello in produzione

## Setup

### Installazione

Evidently AI è già incluso in `requirements.txt`. Se necessario:

```bash
pip install evidently
```

### Configurazione

Configurazione in `configs/config.yaml`:

```yaml
monitoring:
  reference_dataset_path: "data/processed/train.csv"
  reports_dir: "monitoring/reports"
  check_interval_days: 1
  data_drift_threshold: 0.2
  prediction_drift_threshold: 0.15
```

## Data Quality Monitoring

### Scopo

Monitora qualità dati in input rispetto al training set.

### Uso

```python
from src.monitoring.data_quality import generate_data_quality_report
import pandas as pd

# Carica nuovi dati
new_data = pd.read_csv("new_data.csv")

# Genera report
report_path = generate_data_quality_report(
    reference_path="data/processed/train.csv",
    current_data=new_data,
    output_dir="monitoring/reports",
)

print(f"Report salvato: {report_path}")
```

### Metriche Monitorate

- Missing values
- Text length distribution
- Character distribution
- Vocabulary changes

### Alert

Alert generati se:
- Missing values > soglia
- Distribuzione lunghezza testi cambia significativamente
- Vocabolario cambia drasticamente

## Data Drift Detection

### Scopo

Rileva cambiamenti nella distribuzione dei dati input.

### Uso

```python
from src.monitoring.data_drift import check_data_drift
import pandas as pd

# Carica nuovi dati
new_data = pd.read_csv("new_data.csv")

# Controlla drift
drift_results = check_data_drift(
    reference_path="data/processed/train.csv",
    current_data=new_data,
    drift_threshold=0.2,
)

if drift_results["drift_detected"]:
    print("⚠️ Data drift rilevato!")
    print(f"Drift score: {drift_results['drift_score']}")
```

### Metriche

- **PSI** (Population Stability Index)
- **Kolmogorov-Smirnov** test
- Distribuzione features

### Trigger Retraining

Se data drift rilevato:
- Automatico: Retraining FastText
- Manuale: Valutare retraining Transformer

## Prediction Drift Monitoring

### Scopo

Monitora distribuzione delle predizioni nel tempo.

### Uso

```python
from src.monitoring.prediction_drift import monitor_predictions

# Log predizioni (es. da API)
predictions_log = [
    {"prediction": "positive", "timestamp": "2024-01-01"},
    {"prediction": "negative", "timestamp": "2024-01-01"},
    # ...
]

# Distribuzione riferimento (da validation set)
reference_distribution = {
    "positive": 0.4,
    "neutral": 0.3,
    "negative": 0.3,
}

# Monitora
drift_results = monitor_predictions(
    predictions_log=predictions_log,
    reference_distribution=reference_distribution,
)

if drift_results["drift_detected"]:
    print("⚠️ Prediction drift rilevato!")
```

### Metriche

- Distribuzione classi predette
- Confidence scores
- Shift nelle proporzioni

### Alert

Alert se shift > 10% per qualsiasi classe.

## Performance Monitoring

### Scopo

Monitora performance modello se ground truth disponibile.

### Uso

```python
from src.monitoring.performance_monitoring import monitor_performance
import pandas as pd

# Predizioni con ground truth (da feedback)
predictions_with_labels = pd.DataFrame({
    "prediction": ["positive", "negative", ...],
    "label": ["positive", "positive", ...],
})

# Monitora performance
performance = monitor_performance(
    predictions_with_labels=predictions_with_labels,
    reference_path="data/processed/val.csv",
)

print(f"Accuracy: {performance['accuracy']}")
print(f"Macro-F1: {performance['macro_f1']}")
```

### Metriche

- Accuracy
- Macro-F1
- Precision/Recall per classe
- Confusion Matrix

### Degradazione

Se performance degradata:
- Alert generato
- Retraining trigger attivato
- Analisi errori comune

## Dashboard Monitoring

### Avvio Dashboard

```bash
streamlit run src/monitoring/dashboard.py
```

Dashboard disponibile su: `http://localhost:8501`

### Funzionalità

- **Overview**: Statistiche generali
- **Data Quality**: Report qualità dati
- **Data Drift**: Report drift dati
- **Prediction Drift**: Report drift predizioni
- **Performance**: Metriche performance

### Aggiornamento Automatico

Dashboard si aggiorna automaticamente quando nuovi report sono generati.

## Automazione

### Script Schedulato

Crea script per monitoring periodico:

```bash
#!/bin/bash
# monitoring_daily.sh

# Genera report giornalieri
python -m src.monitoring.data_quality
python -m src.monitoring.data_drift
python -m src.monitoring.prediction_drift

# Controlla alert e trigger retraining se necessario
python -m src.training.retrain_fasttext --check-triggers
```

### Cron Job

```bash
# Esegui ogni giorno alle 2 AM
0 2 * * * /path/to/monitoring_daily.sh
```

## Interpretazione Report

### Data Quality

- **Green**: Tutto OK
- **Yellow**: Warning, monitorare
- **Red**: Problema critico, azione richiesta

### Data Drift

- **PSI < 0.1**: Nessun drift
- **PSI 0.1-0.2**: Drift moderato
- **PSI > 0.2**: Drift significativo

### Prediction Drift

- **Shift < 5%**: Normale variazione
- **Shift 5-10%**: Monitorare
- **Shift > 10%**: Drift significativo

## Best Practices

1. **Baseline Solida**: Usa training set come riferimento
2. **Monitoring Continuo**: Controlli giornalieri minimi
3. **Alert Appropriati**: Soglie realistiche per evitare false positive
4. **Documentazione**: Documenta decisioni basate su monitoring
5. **Retraining**: Retrain quando drift significativo rilevato

## Troubleshooting

### Report Non Generati

- Verifica path dataset riferimento
- Controlla formato dati corrente
- Verifica permessi scrittura directory reports

### False Positive Drift

- Aumenta soglie in config
- Verifica qualità dati riferimento
- Considera stagionalità nei dati

### Performance Monitoring Non Disponibile

- Raccogli feedback utenti (`/feedback` endpoint)
- Implementa sistema labeling
- Usa validation set come proxy

