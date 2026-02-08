# Guida Deploy

## Deploy Locale

### Prerequisiti

- Python 3.9+
- Docker (opzionale)
- Modelli addestrati

### Setup Ambiente

```bash
# Clona repository
git clone <repository-url>
cd sentiment-analysis-mlops

# Crea ambiente virtuale
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
pip install -e .
```

### Avvio API Locale

```bash
# Metodo 1: Direttamente
python -m src.api.main

# Metodo 2: Uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Metodo 3: Docker
docker-compose up
```

API disponibile su: `http://localhost:8000`

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Predizione
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Questo è fantastico!", "model_type": "transformer"}'
```

## Deploy Hugging Face Spaces

### Preparazione

1. Crea nuovo Space su Hugging Face
2. Seleziona tipo: **Gradio**
3. Configura repository

### File Necessari

- `app.py`: App Gradio (già presente)
- `requirements.txt`: Dipendenze
- `README.md`: Documentazione Space
- Modelli in `models/` (opzionale, possono essere scaricati)

### Deploy

```bash
# Installa Hugging Face Hub CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Push codice
git push origin main

# Oppure usa web interface di Hugging Face
```

### Configurazione Space

Crea `README.md` per lo Space:

```markdown
---
title: Sentiment Analysis
emoji: 😊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 3.x
app_file: app.py
pinned: false
---

# Sentiment Analysis MLOps

Sistema completo di sentiment analysis con Transformer e FastText.

[Link repository GitHub](https://github.com/yourusername/sentiment-analysis-mlops)
```

## Deploy Docker Production

### Build Immagine

```bash
docker build -t sentiment-analysis-api:latest .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name sentiment-api \
  sentiment-analysis-api:latest
```

### Docker Compose Production

Crea `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

## Deploy Cloud

### AWS (EC2/ECS)

1. Build immagine Docker
2. Push su ECR
3. Deploy su ECS o EC2

### Google Cloud (Cloud Run)

```bash
# Build e push
gcloud builds submit --tag gcr.io/PROJECT_ID/sentiment-api

# Deploy
gcloud run deploy sentiment-api \
  --image gcr.io/PROJECT_ID/sentiment-api \
  --platform managed \
  --region us-central1
```

### Azure (Container Instances)

```bash
# Build e push
az acr build --registry REGISTRY_NAME --image sentiment-api .

# Deploy
az container create \
  --resource-group RESOURCE_GROUP \
  --name sentiment-api \
  --image REGISTRY_NAME.azurecr.io/sentiment-api \
  --cpu 2 \
  --memory 4
```

## Monitoring Setup

### Evidently AI Reports

```bash
# Genera report data quality
python -m src.monitoring.data_quality

# Genera report data drift
python -m src.monitoring.data_drift

# Avvia dashboard
streamlit run src/monitoring/dashboard.py
```

### Logging

I log sono salvati in:
- `logs/sentiment_analysis.log` (applicazione)
- `data/feedback.jsonl` (feedback API)

## Troubleshooting

### Modelli Non Caricati

- Verifica path modelli in `configs/config.yaml`
- Controlla che modelli esistano in `models/`
- Verifica permessi file

### Errori API

- Controlla log: `logs/sentiment_analysis.log`
- Verifica health endpoint: `/health`
- Testa modelli direttamente: `python -c "from src.models.transformer_model import TransformerSentimentModel; m = TransformerSentimentModel()"`

### Performance Lente

- Usa GPU se disponibile
- Riduci batch size
- Usa FastText per inferenza veloce

### Memory Issues

- Riduci `model_cache_size` in config
- Usa un solo modello alla volta
- Aumenta memoria container

## Best Practices

1. **Environment Variables**: Usa `.env` per configurazioni sensibili
2. **Health Checks**: Implementa health checks robusti
3. **Logging**: Log strutturati per debugging
4. **Monitoring**: Setup monitoring prima di produzione
5. **Backup**: Backup modelli e dati regolarmente
6. **Versioning**: Versiona modelli con MLflow
7. **Testing**: Testa in staging prima di produzione

## Scaling

### Orizzontale

- Multiple istanze API con load balancer
- Modelli condivisi su storage network (NFS, S3)

### Verticale

- GPU per Transformer
- Più memoria per batch più grandi
- CPU più veloci per FastText

## Security

- **API Keys**: Non committare chiavi API
- **CORS**: Configura CORS per produzione
- **Rate Limiting**: Implementa rate limiting
- **Input Validation**: Valida sempre input
- **HTTPS**: Usa HTTPS in produzione

