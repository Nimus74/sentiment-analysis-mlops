# Sentiment Analysis MLOps

Sistema focalizzato sull'analisi del sentiment con architettura orientata MLOps, che utilizza un modello Transformer come approccio principale e FastText come baseline addestrata nel progetto per confronto.


## 📋 Overview

Questo progetto implementa un sistema di sentiment analysis che permette di:
- Classificare il sentiment in positivo, neutro, negativo
- Confrontare modelli Transformer e FastText
- Integrare componenti sperimentali di monitoring


## 🏗️ Architettura

Il sistema è composto da:
- **Data Pipeline**: Ingestion, preprocessing, validation e split riproducibili
- **Modelli**: Transformer basato su cardiffnlp/twitter-roberta-base-sentiment-latest come modello principale; FastText addestrato come baseline nel progetto per confronto
- **MLOps**: MLflow per experiment tracking, Evidently AI per monitoring sperimentale
- **API**: FastAPI per inferenza con selezione backend
- **Deploy**: Deploy su Hugging Face opzionale e sperimentale
- **CI/CD**: GitHub Actions con test automatici
- **Monitoring**: Componenti di monitoring sperimentali per data quality e drift
- **Retraining**: Retraining automatico opzionale e sperimentale per FastText


## Allineamento con la traccia della consegna

Sebbene la traccia della consegna menzioni l'uso di FastText, in questo progetto è stato scelto un modello Transformer come soluzione primaria in quanto dimostra prestazioni superiori su testi brevi e rumorosi tipici dei social media. FastText è incluso come baseline supervisionata, addestrata su dataset pubblici e utilizzata per confronto. Questa scelta progettuale è intenzionale e documentata per motivi di accuratezza e completezza nell'analisi.


## Notebook Google Colab (Consegna)

Apri ed esegui il notebook direttamente in Google Colab:

- **Colab**: https://colab.research.google.com/github/Nimus74/sentiment-analysis-mlops/blob/main/notebooks/DELIVERY_colab_sentiment_analysis.ipynb
- **Repository**: https://github.com/Nimus74/sentiment-analysis-mlops

> Nota: in alternativa, il notebook può essere condiviso anche tramite Google Drive (modalità tipica di consegna).


## 🎛 Demo e funzionalità opzionali (non richieste per la consegna)

Oltre al notebook di consegna, il repository include alcune componenti **opzionali** sviluppate per dimostrare un approccio MLOps end-to-end:

### Demo UI – Analisi del Sentiment
È disponibile una semplice interfaccia web (Gradio) per testare l’inferenza dei modelli Transformer e FastText:

```bash
python app.py
```

L’applicazione sarà disponibile all’indirizzo:
http://127.0.0.1:7860

### Monitoring & Reporting (POC)
Il progetto include inoltre una dashboard di monitoring realizzata in Streamlit, con report generati tramite Evidently AI:

```bash
streamlit run src/monitoring/dashboard.py
```

La dashboard consente di visualizzare:
- Data Quality
- Data Drift
- Prediction Drift
- Performance del modello

> Nota: queste funzionalità sono **proof-of-concept** e non sono richieste ai fini della valutazione.
> La consegna ufficiale del progetto è rappresentata dal notebook Google Colab indicato sopra.


## 🚀 Quick Start

### Installazione

```bash
# Clonare il repository
git clone https://github.com/Nimus74/sentiment-analysis-mlops.git
cd sentiment-analysis-mlops

# Creare ambiente virtuale
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installare dipendenze
pip install -r requirements.txt

# Installare package in modalità sviluppo
pip install -e .
```

### Training Modelli

```bash
# Training Transformer
python src/training/train_transformer.py --config configs/config.yaml

# Training FastText
python src/training/train_fasttext.py --config configs/config.yaml
```

### Avviare API

```bash
# Con Docker
docker-compose up

# Oppure direttamente
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Uso API

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "Questo prodotto è fantastico!",
        "model_type": "transformer"  # o "fasttext"
    }
)
print(response.json())
```

## 📊 Metriche

- **Metrica principale**: Macro-F1 Score
- **Metriche secondarie**: Accuracy, Precision, Recall per classe, Confusion Matrix

## 📁 Struttura Progetto

```
sentiment_analysis/
├── data/              # Dataset e cache
├── src/
│   ├── data/         # Data pipeline
│   ├── models/       # Modelli (Transformer, FastText)
│   ├── evaluation/   # Metriche e valutazione
│   ├── api/          # API FastAPI
│   ├── monitoring/   # Evidently AI reports
│   └── training/     # Script training e retraining
├── tests/            # Test unitari e integrazione
├── notebooks/        # Notebook analisi
├── configs/          # File configurazione YAML
├── docs/             # Documentazione
└── .github/workflows/ # CI/CD pipelines
```


## 📚 Documentazione

La documentazione presente nella cartella `docs/` include materiali di supporto e approfondimento
sviluppati durante il progetto, alcuni dei quali in forma di proof-of-concept o documentazione tecnica
di lavoro.

I file principali includono:
- [Guida POC Test Live](docs/POC_TEST_LIVE.md) – guida operativa passo-passo all’esecuzione del progetto
- [Architettura](docs/ARCHITECTURE.md)
- [Modelli](docs/MODELS.md)
- [Deploy](docs/DEPLOYMENT.md)
- [Monitoring](docs/MONITORING.md)

> Nota: parte della documentazione ha carattere **sperimentale o tecnico-interno** ed è fornita
> a supporto della comprensione del progetto.


## 🧪 Testing

```bash
# Eseguire tutti i test
pytest

# Con coverage
pytest --cov=src --cov-report=html
```

## 📝 Stato del progetto / Limitazioni

- CI e test automatici sono implementati e tutti i test sono superati con successo
- Componenti di monitoring sono implementati come proof-of-concept e non ancora integrati in un sistema di produzione completo
- Deploy su Hugging Face e retraining continuo non sono completamente automatizzati e rappresentano estensioni opzionali e sperimentali del progetto
