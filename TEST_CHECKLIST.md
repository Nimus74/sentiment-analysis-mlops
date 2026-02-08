# Checklist Test Completa - Sentiment Analysis MLOps

Questa checklist contiene tutti i test da eseguire per validare completamente il sistema di sentiment analysis.

## 📋 Setup Iniziale

### Ambiente e Dipendenze
- [ ] Verificare che tutte le dipendenze siano installate: `pip install -r requirements.txt`
- [ ] Verificare versione Python >= 3.9: `python3 --version`
- [ ] Verificare che il package sia installato: `pip list | grep sentiment-analysis`
- [ ] Verificare struttura directory: `ls -R src/`

### Configurazione
- [ ] Verificare esistenza `configs/config.yaml`
- [ ] Verificare che tutti i path in config.yaml esistano o siano creabili
- [ ] Verificare che le metriche siano configurate correttamente
- [ ] Verificare che le soglie CI/CD siano ragionevoli

---

## 🔍 Test Data Pipeline

### Download Dataset
- [ ] Eseguire download dataset: `python3 -m src.data.download_dataset`
- [ ] Verificare che il dataset sia stato scaricato in `data/raw/dataset.csv`
- [ ] Verificare esistenza file metadata: `data/raw/dataset_metadata.json`
- [ ] Verificare che il dataset contenga almeno 3000 campioni
- [ ] Verificare distribuzione classi (circa 33% per classe)
- [ ] Verificare che il dataset abbia colonne `text` e `label`

### Preprocessing
- [ ] Eseguire preprocessing: `python3 scripts/prepare_data.py`
- [ ] Verificare che i testi siano stati puliti (nessun URL, menzioni, etc.)
- [ ] Verificare che testi troppo corti siano stati rimossi
- [ ] Verificare che testi troppo lunghi siano stati troncati
- [ ] Verificare esistenza file `data/processed/dataset_processed.csv`
- [ ] Verificare report qualità: `data/processed/quality_report.json`

### Validazione Dati
- [ ] Verificare che non ci siano valori nulli nel dataset processato
- [ ] Verificare che la distribuzione classi sia mantenuta dopo preprocessing
- [ ] Verificare statistiche lunghezza testi nel report qualità
- [ ] Verificare che i duplicati siano stati gestiti

### Split Dati
- [ ] Verificare esistenza file split: `train.csv`, `val.csv`, `test.csv`
- [ ] Verificare che le proporzioni siano corrette (70/15/15)
- [ ] Verificare che la distribuzione classi sia mantenuta in ogni split
- [ ] Verificare esistenza file `data/splits/split_indices.pkl`
- [ ] Verificare esistenza file `data/splits/split_metadata.json`

---

## 🤖 Test Modelli

### Modello Transformer Pre-addestrato
- [ ] Caricare modello pre-addestrato: `python3 -c "from src.models.transformer_model import TransformerSentimentModel; m = TransformerSentimentModel()"`
- [ ] Testare predizione singola: `m.predict("Questo è un test positivo")`
- [ ] Testare predizione batch: `m.predict_batch(["testo1", "testo2"])`
- [ ] Verificare che le label siano corrette (positive/neutral/negative)
- [ ] Verificare che i confidence scores siano tra 0 e 1

### Modello Transformer Fine-tuned
- [ ] Verificare esistenza modello fine-tuned: `ls models/transformer/final_model/`
- [ ] Caricare modello fine-tuned: `python3 -c "from src.models.transformer_model import TransformerSentimentModel; m = TransformerSentimentModel.load('models/transformer/final_model')"`
- [ ] Testare predizione con modello fine-tuned
- [ ] Verificare che le performance siano migliorate rispetto al pre-addestrato

### Modello FastText
- [ ] Verificare esistenza modello FastText: `ls models/fasttext/fasttext_model.bin`
- [ ] Caricare modello FastText: `python3 -c "from src.models.fasttext_model import FastTextSentimentModel; m = FastTextSentimentModel.load('models/fasttext/fasttext_model.bin')"`
- [ ] Testare predizione singola
- [ ] Testare predizione batch
- [ ] Verificare formato label (senza prefisso __label__)

---

## 📊 Test Valutazione e Metriche

### Calcolo Metriche
- [ ] Testare funzione `calculate_metrics()` con dati di esempio
- [ ] Verificare che macro-F1 sia calcolato correttamente
- [ ] Verificare che accuracy sia calcolata correttamente
- [ ] Verificare che precision/recall per classe siano corretti
- [ ] Verificare che confusion matrix sia generata correttamente

### Confronto Modelli
- [ ] Eseguire confronto: `python3 -m src.evaluation.compare_models --config configs/config.yaml`
- [ ] Verificare esistenza report: `reports/model_comparison/comparison_report.txt`
- [ ] Verificare esistenza confusion matrices: `reports/model_comparison/confusion_matrices.png`
- [ ] Verificare che le metriche siano loggate su MLflow
- [ ] Verificare che il confronto mostri differenze significative

### Test Statistici
- [ ] Verificare che i test di significatività siano eseguiti
- [ ] Verificare che i p-values siano calcolati correttamente

---

## 🧪 Test Unitari

### Test Preprocessing
- [ ] Eseguire: `pytest tests/test_preprocessing.py -v`
- [ ] Verificare che tutti i test passino
- [ ] Verificare rimozione URL
- [ ] Verificare rimozione menzioni
- [ ] Verificare normalizzazione hashtag
- [ ] Verificare normalizzazione caratteri speciali

### Test Metriche
- [ ] Eseguire: `pytest tests/test_metrics.py -v`
- [ ] Verificare calcolo metriche base
- [ ] Verificare verifica soglie
- [ ] Verificare confronto metriche

### Test API
- [ ] Eseguire: `pytest tests/test_api.py -v`
- [ ] Verificare health check endpoint
- [ ] Verificare lista modelli endpoint
- [ ] Verificare predizione endpoint (con modelli caricati)

### Test Pipeline
- [ ] Eseguire: `pytest tests/test_pipeline.py -v`
- [ ] Verificare pipeline preprocessing end-to-end
- [ ] Verificare pipeline validazione
- [ ] Verificare pipeline split

### Coverage
- [ ] Eseguire: `pytest --cov=src --cov-report=html`
- [ ] Verificare che coverage sia > 80%
- [ ] Verificare report HTML generato

---

## 🚀 Test API FastAPI

### Avvio API
- [ ] Avviare API: `python3 -m src.api.main` (in background)
- [ ] Verificare che l'API si avvii senza errori
- [ ] Verificare che i modelli vengano caricati correttamente
- [ ] Verificare log di startup

### Endpoint Root
- [ ] Testare: `curl http://localhost:8000/`
- [ ] Verificare risposta JSON corretta
- [ ] Verificare presenza link a `/docs`

### Health Check
- [ ] Testare: `curl http://localhost:8000/health`
- [ ] Verificare status "healthy" o "degraded"
- [ ] Verificare che `models_loaded` mostri stato corretto
- [ ] Verificare che entrambi i modelli siano caricati

### Lista Modelli
- [ ] Testare: `curl http://localhost:8000/models`
- [ ] Verificare che transformer e fasttext siano nella lista
- [ ] Verificare che default_model sia impostato

### Predizione Transformer
- [ ] Testare predizione positiva: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "Questo prodotto è fantastico!", "model_type": "transformer"}'`
- [ ] Verificare risposta con label "positive"
- [ ] Verificare confidence score > 0.5
- [ ] Verificare presenza campo `model_used`

### Predizione FastText
- [ ] Testare predizione con FastText: stesso test con `"model_type": "fasttext"`
- [ ] Verificare risposta corretta
- [ ] Confrontare risultati con Transformer

### Predizione Neutrale e Negativa
- [ ] Testare testo neutro: "Il servizio è stato ok"
- [ ] Testare testo negativo: "Terribile esperienza"
- [ ] Verificare label corrette per entrambi i modelli

### Error Handling
- [ ] Testare testo vuoto: `{"text": "", "model_type": "transformer"}`
- [ ] Verificare errore 422 o gestione corretta
- [ ] Testare modello non valido: `{"text": "test", "model_type": "invalid"}`
- [ ] Verificare errore 503 o 422

### Feedback Endpoint
- [ ] Testare: `curl -X POST http://localhost:8000/feedback -H "Content-Type: application/json" -d '{"text": "test", "prediction": "positive", "actual_label": "positive", "model_used": "transformer"}'`
- [ ] Verificare che il feedback venga salvato
- [ ] Verificare file `data/feedback.jsonl`

### Documentazione API
- [ ] Aprire Swagger UI: `http://localhost:8000/docs`
- [ ] Verificare che tutti gli endpoint siano documentati
- [ ] Testare endpoint direttamente da Swagger UI
- [ ] Verificare esempi di request/response

### Performance API
- [ ] Testare latenza predizione Transformer (dovrebbe essere < 500ms su CPU)
- [ ] Testare latenza predizione FastText (dovrebbe essere < 50ms)
- [ ] Testare throughput con batch di 10 richieste
- [ ] Verificare che l'API gestisca richieste concorrenti

---

## 🐳 Test Docker

### Build Immagine
- [ ] Build immagine: `docker build -t sentiment-analysis-api .`
- [ ] Verificare che il build completi senza errori
- [ ] Verificare dimensione immagine (dovrebbe essere < 5GB)

### Run Container
- [ ] Avviare container: `docker run -d -p 8000:8000 --name test-api sentiment-analysis-api`
- [ ] Verificare che il container si avvii
- [ ] Verificare log container: `docker logs test-api`
- [ ] Testare health check: `curl http://localhost:8000/health`

### Docker Compose
- [ ] Avviare con compose: `docker-compose up -d`
- [ ] Verificare che il servizio sia healthy
- [ ] Testare API attraverso container
- [ ] Verificare volumi montati correttamente

### Cleanup
- [ ] Fermare container: `docker stop test-api`
- [ ] Rimuovere container: `docker rm test-api`
- [ ] Verificare cleanup completo

---

## 📈 Test MLflow

### Setup MLflow
- [ ] Verificare che MLflow sia configurato: `mlflow ui` (opzionale)
- [ ] Verificare esistenza directory `mlruns/`
- [ ] Verificare che gli esperimenti siano tracciati

### Logging Esperimenti
- [ ] Eseguire training FastText e verificare logging su MLflow
- [ ] Eseguire training Transformer e verificare logging
- [ ] Verificare che parametri siano loggati
- [ ] Verificare che metriche siano loggate
- [ ] Verificare che modelli siano salvati come artifacts

### Model Registry
- [ ] Verificare che i modelli siano registrati nel Model Registry
- [ ] Verificare versioning automatico
- [ ] Testare promozione modello a Production

---

## 📊 Test Monitoring Evidently AI

### Data Quality Report
- [ ] Generare report data quality: `python3 -c "from src.monitoring.data_quality import generate_data_quality_report; import pandas as pd; df = pd.read_csv('data/processed/train.csv'); generate_data_quality_report('data/processed/train.csv', df.head(100))"`
- [ ] Verificare esistenza report HTML: `monitoring/reports/data_quality_report.html`
- [ ] Verificare che il report contenga metriche corrette

### Data Drift Detection
- [ ] Generare report data drift: `python3 -c "from src.monitoring.data_drift import check_data_drift; import pandas as pd; df = pd.read_csv('data/processed/train.csv'); check_data_drift('data/processed/train.csv', df.head(100))"`
- [ ] Verificare esistenza report HTML
- [ ] Verificare che drift sia rilevato correttamente (o assente)

### Prediction Drift
- [ ] Creare log predizioni di esempio
- [ ] Generare report prediction drift
- [ ] Verificare distribuzione predizioni

### Dashboard Monitoring
- [ ] Avviare dashboard: `streamlit run src/monitoring/dashboard.py`
- [ ] Verificare che la dashboard si carichi
- [ ] Verificare visualizzazione report Evidently
- [ ] Testare navigazione tra pagine

---

## 🔄 Test Retraining

### Retraining FastText
- [ ] Creare file feedback di esempio: `data/feedback.jsonl`
- [ ] Eseguire retraining: `python3 -m src.training.retrain_fasttext --config configs/config.yaml --force`
- [ ] Verificare che nuovo modello sia creato
- [ ] Verificare che le metriche siano migliorate
- [ ] Verificare promozione modello se migliorato

### Criteri Promozione
- [ ] Testare scenario: nuovo modello migliore → promozione
- [ ] Testare scenario: nuovo modello peggiore → nessuna promozione
- [ ] Verificare rollback automatico se necessario

### Trigger Retraining
- [ ] Simulare data drift e verificare trigger
- [ ] Simulare performance degradation e verificare trigger
- [ ] Verificare schedule temporale (se configurato)

---

## 🔗 Test Integrazione

### Pipeline End-to-End
- [ ] Eseguire pipeline completa:
  1. Download dataset
  2. Preprocessing
  3. Split
  4. Training entrambi i modelli
  5. Valutazione
  6. Confronto
- [ ] Verificare che tutti i passaggi completino senza errori
- [ ] Verificare che i file intermedi siano creati correttamente

### API con Modelli Reali
- [ ] Assicurarsi che modelli siano addestrati
- [ ] Avviare API
- [ ] Testare predizioni con test set reale
- [ ] Verificare che le predizioni siano ragionevoli

### MLflow Integration
- [ ] Verificare che tutti gli esperimenti siano tracciati
- [ ] Verificare che i modelli siano versionati
- [ ] Verificare che le metriche siano confrontabili

---

## 📝 Test Documentazione

### README
- [ ] Verificare che README.md sia completo
- [ ] Verificare che tutti i link funzionino
- [ ] Verificare che gli esempi di codice siano corretti
- [ ] Verificare che le istruzioni di installazione siano chiare

### Documentazione Tecnica
- [ ] Verificare esistenza `docs/ARCHITECTURE.md`
- [ ] Verificare esistenza `docs/MODELS.md`
- [ ] Verificare esistenza `docs/DEPLOYMENT.md`
- [ ] Verificare esistenza `docs/MONITORING.md`
- [ ] Verificare che i diagrammi siano presenti e corretti

### Notebook Colab
- [ ] Verificare che il notebook sia eseguibile
- [ ] Verificare che tutti i link siano presenti
- [ ] Verificare che gli esempi siano chiari
- [ ] Testare esecuzione cella per cella

---

## 🚢 Test Deploy

### Hugging Face Spaces (se configurato)
- [ ] Verificare che `app.py` sia presente
- [ ] Verificare che `requirements.txt` sia completo
- [ ] Testare app Gradio localmente: `gradio app.py`
- [ ] Verificare che l'interfaccia UI funzioni
- [ ] Verificare selezione modello
- [ ] Verificare visualizzazione risultati

### Model Hub (opzionale)
- [ ] Verificare script upload: `scripts/upload_to_hf.py`
- [ ] Verificare model cards siano complete

---

## 🔍 Test CI/CD

### GitHub Actions CI
- [ ] Verificare che `.github/workflows/ci.yml` sia presente
- [ ] Fare commit e push per triggerare CI
- [ ] Verificare che i test vengano eseguiti
- [ ] Verificare che il linting funzioni
- [ ] Verificare che il coverage report sia generato

### Model Evaluation Workflow
- [ ] Verificare che `.github/workflows/model_evaluation.yml` sia presente
- [ ] Creare tag release per triggerare evaluation
- [ ] Verificare che il training venga eseguito
- [ ] Verificare che il gating sulle metriche funzioni
- [ ] Verificare che i modelli siano uploadati come artifacts

---

## 🎯 Test Performance

### Benchmark Latenza
- [ ] Misurare latenza Transformer: 100 predizioni, calcolare media
- [ ] Misurare latenza FastText: 100 predizioni, calcolare media
- [ ] Confrontare risultati (FastText dovrebbe essere 10-100x più veloce)
- [ ] Documentare risultati

### Benchmark Throughput
- [ ] Testare richieste concorrenti (10 simultanee)
- [ ] Misurare throughput (richieste/secondo)
- [ ] Verificare che l'API gestisca il carico

### Benchmark Risorse
- [ ] Misurare uso memoria Transformer
- [ ] Misurare uso memoria FastText
- [ ] Misurare uso CPU durante inferenza
- [ ] Documentare requisiti minimi

---

## 🔐 Test Sicurezza e Robustezza

### Input Validation
- [ ] Testare input molto lunghi (> 1000 caratteri)
- [ ] Testare caratteri speciali e Unicode
- [ ] Testare input vuoto/null
- [ ] Verificare che l'API gestisca errori gracefully

### Error Handling
- [ ] Testare scenario: modello non disponibile
- [ ] Testare scenario: errore durante predizione
- [ ] Verificare che gli errori siano loggati
- [ ] Verificare che gli errori non crashino l'API

### Logging
- [ ] Verificare che i log siano generati: `logs/sentiment_analysis.log`
- [ ] Verificare formato log strutturato
- [ ] Verificare che le predizioni siano tracciate (opzionale)

---

## 📊 Test Risultati Finali

### Confronto Finale Modelli
- [ ] Eseguire confronto completo dopo fine-tuning
- [ ] Verificare che Transformer fine-tuned sia migliore di FastText
- [ ] Verificare che Transformer fine-tuned sia migliore del pre-addestrato
- [ ] Documentare risultati nel report

### Validazione Metriche Business
- [ ] Verificare che macro-F1 > soglia configurata (0.75)
- [ ] Verificare che accuracy sia accettabile (> 0.60)
- [ ] Verificare che non ci siano classi con performance molto bassa (< 0.50 F1)

### Report Finale
- [ ] Verificare che tutti i report siano generati
- [ ] Verificare che le visualizzazioni siano corrette
- [ ] Verificare che i risultati siano riproducibili

---

## ✅ Checklist Finale Pre-Consegna

- [ ] Tutti i test unitari passano
- [ ] Tutti i test integrazione passano
- [ ] API funziona correttamente
- [ ] Modelli sono addestrati e salvati
- [ ] Confronto modelli completato
- [ ] Documentazione completa
- [ ] Notebook Colab funzionante
- [ ] CI/CD configurato e funzionante
- [ ] Monitoring setup (almeno base)
- [ ] Deploy testato (locale o Hugging Face)
- [ ] README aggiornato con risultati
- [ ] Repository GitHub pubblico e completo
- [ ] Tutti i link verificati e funzionanti

---

## 📝 Note per Esecuzione Test

### Ordine Consigliato
1. Setup e configurazione
2. Data pipeline
3. Training modelli
4. Test unitari
5. Test API
6. Test integrazione
7. Test deploy
8. Test performance
9. Validazione finale

### Comandi Rapidi

```bash
# Test completi
pytest tests/ -v --cov=src --cov-report=html

# Pipeline completa
python3 scripts/prepare_data.py
python3 -m src.training.train_fasttext --config configs/config.yaml
python3 -m src.training.train_transformer --config configs/config.yaml --fine-tune
python3 -m src.evaluation.compare_models --config configs/config.yaml

# Test API
python3 -m src.api.main &
python3 scripts/test_api.py

# Monitoring
streamlit run src/monitoring/dashboard.py
```

### Criteri di Successo
- ✅ Tutti i test unitari passano
- ✅ Coverage > 80%
- ✅ API risponde correttamente
- ✅ Modelli hanno performance accettabili (macro-F1 > 0.60)
- ✅ Transformer fine-tuned migliore di FastText
- ✅ Documentazione completa e chiara

---

**Data Creazione**: 2025-01-05  
**Ultimo Aggiornamento**: 2025-01-05  
**Versione**: 1.0

