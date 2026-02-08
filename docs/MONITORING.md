# Monitoring (POC) con Evidently AI

## Overview

Il progetto include un modulo di monitoring come **proof-of-concept** per:
- **Data Quality**: controlli su qualità/struttura dei dati
- **Data Drift**: variazioni nella distribuzione delle feature
- **Prediction Drift**: variazioni nella distribuzione delle classi predette

> Questo monitoring è dimostrativo: non è una pipeline production-ready completa (scheduler/alerting/labeling automatico), ma serve a mostrare come introdurre osservabilità nel sistema. Il monitoring non è integrato in un ambiente di produzione reale ed è pensato esclusivamente a scopo dimostrativo.

## Struttura

- Script/utility: `src/monitoring/`
- Report: `monitoring/reports/`
- Dashboard (POC): `src/monitoring/dashboard.py`

## Avvio rapido

- I dataset utilizzati per il monitoring nel progetto sono simulati a partire dai dati di training/validation.

### 1) Generare report (esempi)

```bash
# Data quality
python -m src.monitoring.data_quality

# Data drift
python -m src.monitoring.data_drift

# Prediction drift
python -m src.monitoring.prediction_drift
```

> I singoli moduli possono richiedere dataset di riferimento e dataset “corrente”. In assenza di dati di produzione, è possibile simulare un current set da subset diversi.

### 2) Dashboard

```bash
streamlit run src/monitoring/dashboard.py
```

Dashboard: `http://localhost:8501`

## Note pratiche

- I report Evidently sono salvati come HTML nella cartella `monitoring/reports/`.
- Il monitoring della **performance** in produzione richiede ground truth (label) raccolte tramite feedback o un processo di labeling; nel progetto è lasciato come possibile estensione.