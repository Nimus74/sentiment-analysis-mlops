# Architettura Sistema Sentiment Analysis (MLOps)

## Overview

Questo progetto implementa una pipeline end-to-end per **sentiment analysis** con un approccio orientato alle pratiche MLOps (data pipeline riproducibile, training/evaluation, API di inferenza, CI). Alcuni componenti (monitoring, deploy su Hugging Face, retraining) sono presenti come **proof-of-concept / estensioni opzionali**.

## Diagramma Architettura

```mermaid
graph TB
    subgraph "Data Pipeline"
        A[Dataset (Hugging Face)] --> B[Download & Validation]
        B --> C[Preprocessing]
        C --> D[Train/Val/Test Split]
    end

    subgraph "Training"
        D --> E[Transformer (baseline pre-trained)]
        D --> F[FastText (supervised)]
        E --> G[MLflow Tracking]
        F --> G
    end

    subgraph "Evaluation"
        G --> H[Model Comparison]
        H --> I[Metrics & Reports]
    end

    subgraph "Serving"
        I --> J[FastAPI Service]
        J --> K[Docker (opzionale)]
    end

    subgraph "Monitoring (POC)"
        J --> M[Evidently AI]
        M --> N[Data Quality]
        M --> O[Data Drift]
        M --> P[Prediction Drift]
        M --> Q[Dashboard]
    end
```

## Componenti Principali

### 1) Data Pipeline (`src/data/`)

- `download_dataset.py`: download dataset (Hugging Face) e controlli di base
- `preprocessing.py`: pulizia testi standardizzata (URL, menzioni, hashtag, ecc.)
- `validation.py`: controlli qualità (colonne attese, duplicati, statistiche descrittive)
- `split.py`: split stratificato riproducibile train/val/test

**Nota**: la pipeline è progettata per applicare lo **stesso preprocessing** a entrambi i modelli, così da rendere il confronto più equo.

### 2) Modelli (`src/models/`)

#### Transformer
- Modello pre-addestrato: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Utilizzato come modello “principale” nel progetto (inference via pipeline HF)

#### FastText
- Modello supervised addestrato sul dataset utilizzato nel progetto
- Utilizzato come baseline leggera e rapida

### 3) Training & Evaluation (`src/training/`, `src/evaluation/`)

- Training script per FastText e gestione esperimenti con MLflow
- Metriche standard: accuracy, precision/recall/F1 (macro/micro/weighted)
- Confronto su stesso split di test

### 4) API Service (`src/api/`)

- Framework: FastAPI
- Endpoint principali:
  - `/predict`: inferenza con selezione modello
  - `/health`: health check
  - `/models`: modelli disponibili
  - `/feedback`: raccolta feedback (POC)

### 5) Monitoring (POC) (`src/monitoring/`)

- Generazione report con Evidently AI (qualità dati e drift)
- Dashboard Streamlit per consultazione report

> Il monitoring è incluso come **POC**: serve a dimostrare la fattibilità dell’osservabilità del modello, non come soluzione production-ready.

## Flusso Dati (alto livello)

1. Download dataset
2. Preprocessing
3. Validazione
4. Split train/val/test
5. Training / inference (Transformer + FastText)
6. Evaluation e confronto
7. Serving via API
8. (Opzionale) report di monitoring

## Scelte Tecnologiche

- **ML / NLP**: Hugging Face Transformers + FastText (baseline)
- **MLOps**: MLflow (tracking)
- **API**: FastAPI
- **Container**: Docker (opzionale)
- **CI**: GitHub Actions (test automatici)
- **Monitoring (POC)**: Evidently AI + Streamlit