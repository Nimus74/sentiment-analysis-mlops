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

Sebbene la traccia della consegna menzioni l'uso di FastText, in questo progetto è stato scelto un modello Transformer come soluzione primaria a causa delle sue prestazioni superiori su testi provenienti dai social media. FastText è incluso come baseline supervisionata, addestrata su dataset pubblici e utilizzata per confronto. Questa scelta progettuale è intenzionale e documentata per motivi di accuratezza e completezza nell'analisi.

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

## 🔗 Link Utili

- **Notebook Colab**: (da completare)
- **Hugging Face Space (deploy opzionale)**: (da completare)
- **Documentazione**: Vedi cartella `docs/`

## 📚 Documentazione

- [Guida POC Test Live](docs/POC_TEST_LIVE.md) - **Guida completa passo-passo per eseguire tutto il progetto**
- [Architettura](docs/ARCHITECTURE.md)
- [Modelli](docs/MODELS.md)
- [Deploy](docs/DEPLOYMENT.md)
- [Monitoring](docs/MONITORING.md)

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
