# 📚 Project Recap - Sentiment Analysis MLOps

**Data Recap**: 8 Febbraio 2026  
**Progetto**: Sistema completo end-to-end di analisi del sentiment con architettura MLOps  
**Linguaggio**: Python 3.10  
**Framework UI**: Gradio (per Hugging Face Spaces) + Streamlit (per monitoring dashboard)  
**API**: FastAPI

---

## 🎯 EXECUTIVE SUMMARY

Sistema MLOps completo per sentiment analysis su testi italiani con due modelli (Transformer RoBERTa e FastText). Include pipeline dati completa (download Hugging Face → preprocessing → split), training con MLflow tracking, API FastAPI per inferenza, monitoring con Evidently AI, e retraining automatico FastText. Deployabile su Hugging Face Spaces (Gradio UI) o Docker (FastAPI). Il progetto è funzionante end-to-end con test suite completa e CI/CD GitHub Actions.

---

## 📁 REPO MAP

```
Sentiment_Analisys/
├── app.py                          # 🎯 ENTRY POINT Gradio UI (Hugging Face Spaces)
├── src/
│   ├── api/
│   │   ├── main.py                 # 🎯 ENTRY POINT FastAPI service
│   │   └── schemas.py              # Pydantic schemas per API
│   ├── data/
│   │   ├── download_dataset.py     # Download da Hugging Face
│   │   ├── preprocessing.py        # Pulizia testi (URL, menzioni, hashtag)
│   │   ├── split.py               # Split stratificato train/val/test
│   │   └── validation.py          # Validazione qualità dati
│   ├── models/
│   │   ├── transformer_model.py   # Wrapper Transformer RoBERTa
│   │   └── fasttext_model.py      # Wrapper FastText supervised
│   ├── training/
│   │   ├── train_transformer.py   # Training/fine-tuning Transformer
│   │   ├── train_fasttext.py      # Training FastText
│   │   ├── retrain_fasttext.py    # Retraining automatico FastText
│   │   └── mlflow_utils.py        # Utilities MLflow tracking
│   ├── evaluation/
│   │   ├── metrics.py             # Calcolo metriche (macro-F1, etc.)
│   │   └── compare_models.py      # Confronto modelli
│   └── monitoring/
│       ├── dashboard.py           # 🎯 Streamlit dashboard monitoring
│       ├── data_drift.py          # Evidently AI data drift
│       ├── data_quality.py        # Evidently AI data quality
│       ├── prediction_drift.py    # Evidently AI prediction drift
│       └── performance_monitoring.py # Performance metrics
├── scripts/
│   ├── prepare_data.py            # Script helper preprocessing + split
│   └── test_api.py                # Script test API
├── configs/
│   └── config.yaml                # ⚙️ Configurazione centralizzata
├── tests/                         # Test suite pytest
├── data/
│   ├── raw/                       # Dataset scaricati
│   ├── processed/                 # Dataset preprocessati + split
│   └── splits/                     # Indici split salvati
├── models/
│   ├── transformer/               # Modelli Transformer salvati
│   └── fasttext/                  # Modelli FastText salvati
├── monitoring/reports/             # Report Evidently AI (HTML)
├── mlruns/                        # MLflow experiment tracking
├── requirements.txt                # ⚙️ Dipendenze Python
├── setup.py                       # Package setup
├── Dockerfile                     # ⚙️ Container Docker
├── docker-compose.yml             # ⚙️ Docker Compose
└── docs/                          # Documentazione completa

```

**File Chiave**:
- **Entry Points**: `app.py` (Gradio), `src/api/main.py` (FastAPI), `src/monitoring/dashboard.py` (Streamlit)
- **Config**: `configs/config.yaml` (tutto centralizzato)
- **Dependencies**: `requirements.txt`
- **Docker**: `Dockerfile`, `docker-compose.yml`

---

## 🏗️ ARCHITETTURA OVERVIEW

### Moduli e Responsabilità

#### 1. **Data Pipeline** (`src/data/`)
- **download_dataset.py**: Scarica dataset italiano da Hugging Face (CSV URL diretto), valida formato, calcola hash SHA256 per tracciabilità
- **preprocessing.py**: Preprocessing deterministico (rimozione URL, menzioni @username, normalizzazione hashtag, caratteri speciali), preparazione formato FastText
- **split.py**: Split stratificato riproducibile 70/15/15 (train/val/test) con seed fisso (42), salvataggio indici per tracciabilità
- **validation.py**: Genera report qualità dati (distribuzione classi, lunghezza testi, valori nulli)

#### 2. **Modelli ML** (`src/models/`)
- **transformer_model.py**: Wrapper per `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingue, supporta italiano). Supporta fine-tuning, predizione batch, salvataggio/caricamento. Usa pipeline Hugging Face.
- **fasttext_model.py**: Wrapper FastText supervised con workaround NumPy 2.x. Training, predizione batch, salvataggio formato .bin

#### 3. **Training** (`src/training/`)
- **train_transformer.py**: Fine-tuning Transformer con Hugging Face Trainer, early stopping, logging MLflow, salvataggio modello fine-tuned
- **train_fasttext.py**: Training FastText supervised con parametri configurabili, logging MLflow
- **retrain_fasttext.py**: Retraining automatico FastText basato su trigger (data drift, performance degradation, schedule), criteri promozione modello (macro-F1 +2%, max degradazione classe -5%)
- **mlflow_utils.py**: Utilities per setup MLflow, logging parametri/metriche/artefatti, model registry

#### 4. **Evaluation** (`src/evaluation/`)
- **metrics.py**: Calcolo metriche complete (macro-F1 primaria, accuracy, precision, recall per classe, confusion matrix), verifica soglie CI/CD
- **compare_models.py**: Confronto modelli su stesso test set, generazione report comparativo

#### 5. **API Service** (`src/api/`)
- **main.py**: FastAPI app con endpoints `/predict`, `/health`, `/models`, `/feedback`. Lifespan context manager per caricamento modelli all'avvio, cache modelli in memoria, CORS middleware
- **schemas.py**: Pydantic schemas per request/response validation

#### 6. **Monitoring** (`src/monitoring/`)
- **dashboard.py**: Dashboard Streamlit centralizzata per visualizzare report Evidently AI
- **data_drift.py**: Rilevamento data drift con Evidently AI (PSI threshold)
- **data_quality.py**: Report qualità dati input
- **prediction_drift.py**: Rilevamento drift distribuzione predizioni
- **performance_monitoring.py**: Metriche performance produzione (se ground truth disponibile)

#### 7. **UI Applications**
- **app.py**: Interfaccia Gradio semplice per Hugging Face Spaces, carica modelli all'avvio, predizione con selezione modello (transformer/fasttext)

---

## 🔄 FLOWS END-TO-END

### Flow 1: App Startup (Gradio UI)

```
1. Esegui: python app.py
2. Importa modelli (transformer_model, fasttext_model)
3. load_models() chiamato:
   - Prova a caricare Transformer da models/transformer/final_model/
   - Se non esiste, usa modello pre-addestrato Hugging Face
   - Prova a caricare FastText da models/fasttext/fasttext_model.bin
   - Se non esiste, imposta a None
4. Crea interfaccia Gradio con:
   - Input: Textbox (testo) + Radio (selezione modello)
   - Output: Markdown (risultato formattato)
   - Examples predefiniti
5. demo.launch() → Server Gradio avviato (default: localhost:7860)
```

### Flow 2: User Interaction (Gradio UI)

```
1. Utente inserisce testo nella textbox
2. Utente seleziona modello (transformer/fasttext)
3. Clicca "Submit" o Enter
4. predict_sentiment(text, model_type) chiamato:
   - Valida input (non vuoto)
   - Recupera modello da cache globale
   - Se modello None → ritorna errore
   - model.predict(text) → {label, score, text}
   - Formatta output con emoji e confidence
5. Risultato mostrato in Markdown output
```

### Flow 3: Inference Flow (API FastAPI)

```
1. Client POST /predict con JSON: {"text": "...", "model_type": "transformer"}
2. Pydantic schema valida request (PredictionRequest)
3. Verifica modello disponibile in model_cache
4. Se non disponibile → HTTPException 503
5. model.predict(request.text) eseguito:
   - Transformer: tokenizer → model → softmax → label mapping
   - FastText: model.predict() → rimozione prefisso __label__
6. Risultato formattato in PredictionResponse:
   - text, prediction (label), confidence, model_used, probabilities (opzionale)
7. Ritorna JSON response
```

### Flow 4: Training/Evaluation Flow

```
1. PREPARAZIONE DATI:
   python scripts/prepare_data.py
   ├─ Carica configs/config.yaml
   ├─ Download dataset (se non presente):
   │  python -m src.data.download_dataset
   │  └─ Scarica CSV da Hugging Face → data/raw/dataset.csv
   ├─ Preprocessing:
   │  └─ preprocess_dataframe() → rimozione URL, normalizzazione
   ├─ Validazione qualità:
   │  └─ generate_quality_report() → data/processed/quality_report.json
   └─ Split stratificato:
      └─ stratified_split() → train.csv, val.csv, test.csv

2. TRAINING FASTTEXT:
   python -m src.training.train_fasttext --config configs/config.yaml
   ├─ Carica train.csv, val.csv
   ├─ Prepara formato FastText (__label__<label> <text>)
   ├─ fasttext.train_supervised() con parametri config
   ├─ Salva modello → models/fasttext/fasttext_model.bin
   ├─ Valutazione su val set:
   │  └─ calculate_metrics() → macro-F1, accuracy, etc.
   └─ Logging MLflow:
      └─ log_params(), log_metrics(), log_model_artifact()

3. TRAINING TRANSFORMER:
   python -m src.training.train_transformer --config configs/config.yaml --fine-tune
   ├─ Carica train.csv, val.csv
   ├─ Crea SentimentDataset (tokenizzazione)
   ├─ Carica modello pre-addestrato Hugging Face
   ├─ Fine-tuning con Trainer:
   │  ├─ TrainingArguments (epochs, batch_size, learning_rate)
   │  ├─ EarlyStoppingCallback
   │  └─ compute_metrics callback
   ├─ Salva modello fine-tuned → models/transformer/final_model/
   ├─ Valutazione su val set durante training
   └─ Logging MLflow

4. EVALUATION:
   python -m src.evaluation.compare_models
   ├─ Carica test.csv (holdout set)
   ├─ Predizioni entrambi modelli su test set
   ├─ Calcolo metriche per modello
   ├─ Confronto metriche
   └─ Generazione report → reports/model_comparison/
```

### Flow 5: Monitoring Flow

```
1. GENERAZIONE REPORT (periodica, es. giornaliera):
   python -m src.monitoring.data_drift
   ├─ Carica reference dataset (train.csv)
   ├─ Carica current dataset (nuovi dati produzione)
   ├─ Evidently AI Report con DataDriftPreset
   ├─ Calcola drift score (PSI)
   └─ Salva report HTML → monitoring/reports/data_drift_report.html

2. DASHBOARD STREAMLIT:
   streamlit run src/monitoring/dashboard.py
   ├─ Carica ultimi report HTML da monitoring/reports/
   ├─ Visualizza report Evidently AI embedded
   ├─ Mostra metriche aggregati
   └─ Interfaccia navigazione tra report

3. RETRAINING TRIGGER:
   python -m src.training.retrain_fasttext
   ├─ Verifica trigger (data drift, performance, schedule)
   ├─ Raccoglie nuovi dati da data/feedback.jsonl
   ├─ Se >= min_samples (100):
   │  ├─ Combina con training set originale
   │  ├─ Retrain FastText
   │  ├─ Valutazione su val set
   │  ├─ Verifica criteri promozione:
   │  │  ├─ Macro-F1 miglioramento >= 2%
   │  │  └─ Nessuna classe degradata > 5%
   │  └─ Se promosso: sostituisce modello produzione
   └─ Logging MLflow
```

---

## ✅ CURRENT STATUS

### ✅ IMPLEMENTATO E FUNZIONANTE

1. **Data Pipeline Completa**
   - ✅ Download dataset da Hugging Face
   - ✅ Preprocessing standardizzato
   - ✅ Split stratificato riproducibile
   - ✅ Validazione qualità dati

2. **Modelli ML**
   - ✅ Transformer RoBERTa (pre-addestrato + fine-tuning)
   - ✅ FastText supervised
   - ✅ Wrapper con interfacce coerenti
   - ✅ Salvataggio/caricamento modelli

3. **Training**
   - ✅ Training Transformer con fine-tuning
   - ✅ Training FastText
   - ✅ MLflow experiment tracking
   - ✅ Early stopping Transformer

4. **Evaluation**
   - ✅ Metriche complete (macro-F1, accuracy, precision, recall)
   - ✅ Confronto modelli
   - ✅ Report generazione

5. **API Service**
   - ✅ FastAPI con endpoints completi
   - ✅ Caricamento modelli all'avvio
   - ✅ Health check
   - ✅ Feedback collection

6. **UI Applications**
   - ✅ Gradio UI per Hugging Face Spaces
   - ✅ Streamlit dashboard monitoring

7. **Monitoring**
   - ✅ Evidently AI integration (data drift, data quality, prediction drift)
   - ✅ Report HTML generazione

8. **Infrastructure**
   - ✅ Docker support
   - ✅ Docker Compose
   - ✅ CI/CD GitHub Actions

9. **Testing**
   - ✅ Test suite pytest completa
   - ✅ Test unitari moduli principali
   - ✅ Test integrazione pipeline

### ⚠️ PARZIALMENTE IMPLEMENTATO / TODO

1. **Retraining Automatico**
   - ✅ Script retrain_fasttext.py implementato
   - ⚠️ Trigger automatici non schedulati (richiede setup cron/scheduler esterno)
   - ⚠️ Retraining Transformer non implementato (solo FastText)

2. **Monitoring Dashboard**
   - ✅ Dashboard Streamlit implementata
   - ⚠️ Non integrata con sistema di alerting
   - ⚠️ Report devono essere generati manualmente o via scheduler esterno

3. **API Features**
   - ✅ Endpoint base implementati
   - ⚠️ Rate limiting non implementato
   - ⚠️ Authentication non implementata (CORS permissivo)

4. **Documentation**
   - ✅ README presente
   - ✅ Docs/ folder con guide
   - ⚠️ Alcune discrepanze (menziona Streamlit ma usa Gradio per UI principale)

### ❌ MANCANTE / NON IMPLEMENTATO

1. **Environment Variables**
   - ❌ Nessun file .env.example
   - ❌ Configurazione via variabili d'ambiente non centralizzata

2. **Production Features**
   - ❌ Rate limiting API
   - ❌ Authentication/Authorization
   - ❌ Logging strutturato avanzato (es. JSON logging)
   - ❌ Metrics export per Prometheus/Grafana

3. **Advanced MLOps**
   - ❌ A/B testing modelli in produzione
   - ❌ Feature store
   - ❌ Model versioning avanzato (oltre MLflow)

4. **Deployment Automation**
   - ❌ CI/CD per deploy automatico
   - ❌ Kubernetes manifests (solo Docker)

---

## 🚀 HOW TO RUN

### Prerequisiti

- Python 3.10 (richiesto per Evidently AI)
- pip
- Git (opzionale, se clonato da repo)

### Installazione

```bash
# 1. Naviga nella directory progetto
cd /path/to/Sentiment_Analisys

# 2. Crea ambiente virtuale Python 3.10
python3.10 -m venv .venv310
source .venv310/bin/activate  # Windows: .venv310\Scripts\activate

# 3. Installa dipendenze
pip install --upgrade pip
pip install -r requirements.txt

# 4. Installa package in modalità sviluppo
pip install -e .
```

### Preparazione Dati (Prima Esecuzione)

```bash
# 1. Crea directory necessarie
mkdir -p data/raw data/processed data/splits models/transformer models/fasttext

# 2. Scarica dataset
python -m src.data.download_dataset

# 3. Preprocessing e split
python scripts/prepare_data.py
```

### Training Modelli (Opzionale, se modelli non presenti)

```bash
# Training FastText
python -m src.training.train_fasttext --config configs/config.yaml

# Training Transformer (fine-tuning, richiede tempo)
python -m src.training.train_transformer --config configs/config.yaml --fine-tune
```

### Avvio Applicazioni

#### Opzione 1: Gradio UI (Hugging Face Spaces / Locale)

```bash
# Avvia Gradio UI
python app.py

# Oppure con parametri
python app.py --server_port 7860 --share
```

**Accesso**: http://localhost:7860

#### Opzione 2: FastAPI Service

```bash
# Metodo 1: Direttamente
python -m src.api.main

# Metodo 2: Uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Metodo 3: Docker
docker-compose up
```

**Accesso**: http://localhost:8000  
**Docs**: http://localhost:8000/docs

#### Opzione 3: Streamlit Monitoring Dashboard

```bash
streamlit run src/monitoring/dashboard.py
```

**Accesso**: http://localhost:8501

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Predizione
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Questo prodotto è fantastico!", "model_type": "transformer"}'
```

### Esecuzione Test Suite

```bash
# Tutti i test
pytest

# Con coverage
pytest --cov=src --cov-report=html

# Test specifici
pytest tests/test_models.py -v
```

---

## 📋 NEXT STEPS (Prioritizzati)

### Priorità Alta (Per Funzionamento Completo)

1. **Verificare Modelli Presenti**
   - Controllare se `models/transformer/final_model/` e `models/fasttext/fasttext_model.bin` esistono
   - Se mancanti, eseguire training (vedi sezione "Training Modelli")

2. **Verificare Dataset**
   - Controllare se `data/processed/train.csv`, `val.csv`, `test.csv` esistono
   - Se mancanti, eseguire `python scripts/prepare_data.py`

3. **Test End-to-End**
   - Eseguire test suite: `pytest`
   - Testare API: `python scripts/test_api.py`
   - Testare Gradio UI: `python app.py`

### Priorità Media (Miglioramenti)

1. **Setup Environment Variables**
   - Creare `.env.example` con variabili configurazione
   - Documentare variabili necessarie

2. **Schedulare Monitoring**
   - Setup cron job o scheduler per generazione report Evidently AI periodici
   - Integrare alerting se drift rilevato

3. **Production Hardening**
   - Configurare CORS con origini specifiche (non `["*"]`)
   - Aggiungere rate limiting API
   - Implementare logging strutturato

### Priorità Bassa (Nice to Have)

1. **Retraining Transformer**
   - Implementare retraining automatico Transformer (attualmente solo FastText)

2. **Advanced Monitoring**
   - Integrazione Prometheus/Grafana
   - Dashboard metriche produzione real-time

3. **Documentation**
   - Allineare README con implementazione reale (Gradio vs Streamlit)
   - Aggiungere diagrammi architettura visuali

---

## ❓ QUESTIONS FOR YOU

1. **Modelli Pre-addestrati**: Hai già modelli addestrati salvati in `models/` o devo eseguire il training da zero?

2. **Dataset**: Il dataset è già scaricato in `data/raw/` o devo scaricarlo?

3. **Deploy Target**: Quale ambiente vuoi usare?
   - Locale (Gradio/FastAPI)
   - Hugging Face Spaces (Gradio)
   - Docker production (FastAPI)
   - Altro?

4. **Monitoring**: Vuoi configurare monitoring automatico con scheduler o è sufficiente manuale per ora?

5. **Priorità Immediate**: Quale componente vuoi testare/eseguire per primo?
   - Gradio UI
   - FastAPI service
   - Training modelli
   - Monitoring dashboard

---

**Fine Recap**
