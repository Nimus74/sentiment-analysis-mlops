# Sentiment Analysis MLOps

Sistema completo end-to-end di analisi del sentiment sui social media con architettura MLOps, confronto tra modelli Transformer e FastText, deploy su Hugging Face, monitoring continuo con Evidently AI e retraining automatico.

## 📋 Overview

Questo progetto implementa un sistema di sentiment analysis che permette di:
- Analizzare automaticamente testi provenienti dai social media
- Classificare il sentiment in positivo, neutro, negativo
- Monitorare nel tempo l'andamento della reputazione
- Adattarsi a cambiamenti nel linguaggio tramite retraining automatico

## 🏗️ Architettura

Il sistema è composto da:
- **Data Pipeline**: Ingestion, preprocessing, validation e split riproducibili
- **Modelli**: Transformer (cardiffnlp/twitter-roberta-base-sentiment-latest) e FastText supervised
- **MLOps**: MLflow per experiment tracking, Evidently AI per monitoring
- **API**: FastAPI per inferenza con selezione backend
- **Deploy**: Hugging Face Spaces e Model Hub
- **CI/CD**: GitHub Actions con test automatici e gating metriche
- **Monitoring**: Data quality, data drift, prediction drift
- **Retraining**: Automatico per FastText con criteri di promozione

## 🚀 Quick Start

### Installazione

```bash
# Clonare il repository
git clone https://github.com/yourusername/sentiment-analysis-mlops.git
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
- **Soglia CI/CD**: Macro-F1 > 0.75

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

- **Notebook Colab**: [Link al notebook](https://colab.research.google.com/...)
- **Hugging Face Space**: [Link allo Space](https://huggingface.co/spaces/...)
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

## 🤝 Contributing

Contributi benvenuti! Per maggiori dettagli vedi [CONTRIBUTING.md](CONTRIBUTING.md).

## 📝 License

MIT License - vedi [LICENSE](LICENSE) per dettagli.

## 👥 Team

AI Engineering & MLOps Team

