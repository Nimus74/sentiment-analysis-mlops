# 🚀 Guida Completa POC Test Live - macOS

**Versione**: 1.0  
**Data**: 2025-01-05  
**Sistema**: macOS  
**Python**: 3.10 (richiesto per Evidently AI)

---

## 📋 Prerequisiti

- macOS con Python 3.10 installato
- Git installato
- Terminale macOS
- Connessione internet per download dataset e modelli

---

## STEP 1: Setup Ambiente

### 1.1 Clonare/Navigare nel progetto

```bash
cd /Users/francescoscarano/Desktop/Progetto_Mod_5/Sentiment_Analisys
```

### 1.2 Creare ambiente virtuale Python 3.10

```bash
# Installa Python 3.10 se non presente
brew install python@3.10

# Crea ambiente virtuale
python3.10 -m venv .venv310

# Attiva ambiente
source .venv310/bin/activate
```

### 1.3 Verificare versione Python

```bash
python --version
# Dovrebbe mostrare: Python 3.10.x
```

### 1.4 Installare dipendenze

```bash
# Aggiorna pip
pip install --upgrade pip

# Installa dipendenze
pip install -r requirements.txt

# Installa progetto in modalità sviluppo
pip install -e .
```

### 1.5 Verificare installazione

```bash
# Verifica package installato
pip list | grep sentiment-analysis

# Verifica moduli principali
python -c "from src.models.transformer_model import TransformerSentimentModel; print('✅ Transformer OK')"
python -c "from src.models.fasttext_model import FastTextSentimentModel; print('✅ FastText OK')"
python -c "import evidently; print(f'✅ Evidently {evidently.__version__} OK')"
python -c "import mlflow; print(f'✅ MLflow {mlflow.__version__} OK')"
```

**Output atteso**: Tutti i moduli devono essere importati senza errori.

---

## STEP 2: Preparazione Dati

### 2.1 Creare directory necessarie

```bash
mkdir -p data/raw data/processed data/splits models/transformer models/fasttext reports/model_comparison monitoring/reports logs
```

### 2.2 Scaricare dataset

```bash
python -m src.data.download_dataset
```

**Verifica**:
```bash
# Verifica dataset scaricato
ls -lh data/raw/dataset.csv
cat data/raw/dataset_metadata.json

# Verifica struttura
python -c "import pandas as pd; df = pd.read_csv('data/raw/dataset.csv'); print(f'Campioni: {len(df)}'); print(f'Colonne: {df.columns.tolist()}'); print(f'Distribuzione classi:\n{df[\"label\"].value_counts()}')"
```

**Output atteso**: Dataset con almeno 3000 campioni, colonne `text` e `label`, distribuzione classi bilanciata.

### 2.3 Preprocessing e split dati

```bash
python scripts/prepare_data.py
```

**Verifica**:
```bash
# Verifica file creati
ls -lh data/processed/*.csv
ls -lh data/splits/*.csv

# Verifica distribuzione split
python -c "import pandas as pd; train=pd.read_csv('data/splits/train.csv'); val=pd.read_csv('data/splits/val.csv'); test=pd.read_csv('data/splits/test.csv'); print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}'); print(f'Distribuzione train:\n{train[\"label\"].value_counts(normalize=True)}')"
```

**Output atteso**: File train.csv, val.csv, test.csv creati con proporzioni 70/15/15.

---

## STEP 3: Training Modelli

### 3.1 Training FastText

```bash
python -m src.training.train_fasttext --config configs/config.yaml
```

**Verifica**:
```bash
# Verifica modello creato
ls -lh models/fasttext/fasttext_model.bin

# Test caricamento
python -c "from src.models.fasttext_model import FastTextSentimentModel; m = FastTextSentimentModel.load('models/fasttext/fasttext_model.bin'); print('✅ FastText caricato'); print(m.predict('Questo è un test positivo'))"
```

**Output atteso**: Modello FastText creato (~100MB), predizione funzionante.

### 3.2 Training Transformer (fine-tuning)

```bash
python -m src.training.train_transformer --config configs/config.yaml --fine-tune
```

**⚠️ Nota**: Questo step può richiedere tempo (10-30 minuti su CPU).

**Verifica**:
```bash
# Verifica modello creato
ls -lh models/transformer/final_model/

# Test caricamento
python -c "from src.models.transformer_model import TransformerSentimentModel; m = TransformerSentimentModel.load('models/transformer/final_model'); print('✅ Transformer caricato'); print(m.predict('Questo è un test positivo'))"
```

**Output atteso**: Modello Transformer fine-tuned creato, predizione funzionante.

---

## STEP 4: Valutazione e Confronto Modelli

### 4.1 Confronto modelli

```bash
python -m src.evaluation.compare_models --config configs/config.yaml
```

**Verifica**:
```bash
# Verifica report generati
ls -lh reports/model_comparison/

# Visualizza report testuale
cat reports/model_comparison/comparison_report.txt

# Visualizza confusion matrices
open reports/model_comparison/confusion_matrices.png
```

**Output atteso**: Report di confronto con metriche (macro-F1, accuracy, etc.) e confusion matrices.

---

## STEP 5: Test Unitari e Coverage

### 5.1 Eseguire test unitari

```bash
# Tutti i test
pytest tests/ -v

# Test specifici
pytest tests/test_preprocessing.py -v
pytest tests/test_metrics.py -v
pytest tests/test_api.py -v
pytest tests/test_models.py -v
pytest tests/test_validation.py -v
pytest tests/test_split.py -v
pytest tests/test_monitoring.py -v
```

**Output atteso**: Tutti i test devono passare (o almeno la maggior parte).

### 5.2 Generare coverage report

```bash
# Coverage con report HTML e terminale
pytest --cov=src --cov-report=html --cov-report=term-missing

# Visualizza report HTML
open htmlcov/index.html
```

**Verifica coverage**:
```bash
# Verifica percentuale coverage
pytest --cov=src --cov-report=term | grep TOTAL
```

**Output atteso**: Coverage > 80% (obiettivo).

---

## STEP 6: MLflow Tracking

### 6.1 Avviare MLflow UI

```bash
# In un nuovo terminale (mantieni il primo attivo)
source .venv310/bin/activate
mlflow ui --backend-store-uri file:./mlruns --port 5000
```

**Accesso**: Apri browser su `http://localhost:5000`

### 6.2 Verificare esperimenti MLflow

```bash
# Verifica esperimenti
python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
exps = mlflow.search_experiments()
print(f'Esperimenti trovati: {len(exps)}')
for exp in exps:
    print(f'  - {exp.name} (ID: {exp.experiment_id})')
"

# Verifica run
python -c "
import mlflow
mlflow.set_tracking_uri('file:./mlruns')
exp = mlflow.get_experiment_by_name('sentiment_analysis')
if exp:
    runs = mlflow.search_runs([exp.experiment_id], max_results=5)
    print('Ultimi run:')
    print(runs[['tags.mlflow.runName', 'metrics.macro_f1', 'metrics.accuracy']].head())
"
```

**Output atteso**: Esperimenti e run visibili su MLflow UI con metriche loggate.

---

## STEP 7: API FastAPI

### 7.1 Avviare API

```bash
# In un nuovo terminale
source .venv310/bin/activate
cd /Users/francescoscarano/Desktop/Progetto_Mod_5/Sentiment_Analisys
python -m src.api.main
```

**API disponibile su**: `http://localhost:8000`

### 7.2 Test API

```bash
# Health check
curl http://localhost:8000/health | python -m json.tool

# Lista modelli
curl http://localhost:8000/models | python -m json.tool

# Predizione Transformer
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Questo prodotto è fantastico!", "model_type": "transformer"}' \
  | python -m json.tool

# Predizione FastText
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Questo prodotto è fantastico!", "model_type": "fasttext"}' \
  | python -m json.tool

# Test con testo negativo
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terribile esperienza, non lo consiglio.", "model_type": "transformer"}' \
  | python -m json.tool

# Test con testo neutro
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Il servizio è stato ok, niente di speciale.", "model_type": "transformer"}' \
  | python -m json.tool
```

**Output atteso**: API risponde correttamente con predizioni e confidence scores.

### 7.3 Documentazione API (Swagger UI)

Apri nel browser: `http://localhost:8000/docs`

### 7.4 Verificare log API

```bash
# Visualizza log
tail -f logs/sentiment_analysis.log

# Oppure ultime 50 righe
tail -n 50 logs/sentiment_analysis.log
```

**Output atteso**: Log scritti correttamente su file con formato strutturato.

---

## STEP 8: Monitoring Evidently AI

### 8.1 Generare report Data Quality

```bash
python -c "
from src.monitoring.data_quality import generate_data_quality_report
import pandas as pd

# Carica dati di riferimento
ref_df = pd.read_csv('data/processed/train.csv')

# Simula dati correnti (usa un subset)
current_df = ref_df.head(100)

# Genera report
report_path = generate_data_quality_report(
    reference_path='data/processed/train.csv',
    current_data=current_df,
    output_dir='monitoring/reports',
    report_name='data_quality_report.html'
)
print(f'✅ Report salvato: {report_path}')
"
```

### 8.2 Generare report Data Drift

```bash
python -c "
from src.monitoring.data_drift import check_data_drift
import pandas as pd

# Carica dati di riferimento
ref_df = pd.read_csv('data/processed/train.csv')

# Simula dati correnti con drift (usa subset diverso)
current_df = ref_df.tail(100)

# Controlla drift
drift_results = check_data_drift(
    reference_path='data/processed/train.csv',
    current_data=current_df,
    drift_threshold=0.2,
    output_dir='monitoring/reports',
    report_name='data_drift_report.html'
)

print(f'Drift rilevato: {drift_results[\"drift_detected\"]}')
print(f'Drift score: {drift_results.get(\"drift_score\", 0.0)}')
"
```

### 8.3 Generare report Prediction Drift

```bash
python -c "
from src.monitoring.prediction_drift import check_prediction_drift
import pandas as pd

# Carica dati con predizioni (simula da test set)
test_df = pd.read_csv('data/splits/test.csv')

# Simula predizioni (in produzione verrebbero da API)
# Per il test, usiamo le label come proxy delle predizioni
test_df['prediction'] = test_df['label']

# Crea reference e current
reference_df = test_df.head(200)
current_df = test_df.tail(100)

# Controlla prediction drift
drift_results = check_prediction_drift(
    reference_data=reference_df,
    current_data=current_df,
    output_dir='monitoring/reports',
    report_name='prediction_drift_report.html'
)

print(f'Prediction drift rilevato: {drift_results[\"drift_detected\"]}')
print(f'Drift score: {drift_results.get(\"drift_score\", 0.0)}')
print(f'Distribuzione reference: {drift_results.get(\"reference_distribution\", {})}')
print(f'Distribuzione current: {drift_results.get(\"current_distribution\", {})}')
"
```

### 8.4 Verificare report generati

```bash
# Lista report
ls -lh monitoring/reports/*.html

# Apri report nel browser
open monitoring/reports/data_quality_report.html
open monitoring/reports/data_drift_report.html
open monitoring/reports/prediction_drift_report.html
```

**Output atteso**: Report HTML generati correttamente (3-4MB ciascuno).

---

## STEP 9: Dashboard Streamlit

### 9.1 Avviare dashboard

```bash
# In un nuovo terminale
source .venv310/bin/activate
cd /Users/francescoscarano/Desktop/Progetto_Mod_5/Sentiment_Analisys
streamlit run src/monitoring/dashboard.py
```

**Accesso**: Apri browser su `http://localhost:8501`

### 9.2 Navigare dashboard

La dashboard include:
- **Overview**: Panoramica report disponibili
- **Data Quality**: Report qualità dati
- **Data Drift**: Report drift dati
- **Prediction Drift**: Report drift predizioni
- **Performance**: Report performance modelli

**Output atteso**: Dashboard visualizza tutti i report Evidently AI generati.

---

## STEP 10: Verifica Finale

### 10.1 Checklist verifica

```bash
# 1. Verifica modelli addestrati
echo "=== Modelli ==="
ls -lh models/transformer/final_model/ 2>/dev/null && echo "✅ Transformer" || echo "❌ Transformer"
ls -lh models/fasttext/fasttext_model.bin 2>/dev/null && echo "✅ FastText" || echo "❌ FastText"

# 2. Verifica dataset
echo -e "\n=== Dataset ==="
[ -f "data/splits/train.csv" ] && echo "✅ Train split" || echo "❌ Train split"
[ -f "data/splits/val.csv" ] && echo "✅ Val split" || echo "❌ Val split"
[ -f "data/splits/test.csv" ] && echo "✅ Test split" || echo "❌ Test split"

# 3. Verifica report
echo -e "\n=== Report ==="
[ -f "reports/model_comparison/comparison_report.txt" ] && echo "✅ Comparison report" || echo "❌ Comparison report"
[ -f "monitoring/reports/data_quality_report.html" ] && echo "✅ Data quality report" || echo "❌ Data quality report"
[ -f "monitoring/reports/data_drift_report.html" ] && echo "✅ Data drift report" || echo "❌ Data drift report"
[ -f "monitoring/reports/prediction_drift_report.html" ] && echo "✅ Prediction drift report" || echo "❌ Prediction drift report"

# 4. Verifica MLflow
echo -e "\n=== MLflow ==="
[ -d "mlruns" ] && echo "✅ MLflow runs" || echo "❌ MLflow runs"

# 5. Verifica log
echo -e "\n=== Log ==="
[ -f "logs/sentiment_analysis.log" ] && echo "✅ API log" || echo "❌ API log"

# 6. Verifica coverage
echo -e "\n=== Coverage ==="
[ -d "htmlcov" ] && echo "✅ Coverage report" || echo "❌ Coverage report"
```

### 10.2 Test end-to-end

```bash
# Script test completo
python scripts/test_api.py
```

**Output atteso**: Tutti i test API passano correttamente.

---

## STEP 11: Visualizzazione Risultati

### 11.1 MLflow UI

- **URL**: `http://localhost:5000`
- **Visualizza**: Esperimenti, metriche, parametri, artifacts

### 11.2 Coverage Report HTML

```bash
open htmlcov/index.html
```

### 11.3 Report Evidently AI

```bash
# Apri tutti i report
open monitoring/reports/*.html
```

### 11.4 Dashboard Streamlit

- **URL**: `http://localhost:8501`
- **Visualizza**: Tutti i report di monitoring

---

## 📝 Comandi Rapidi Riepilogo

```bash
# Setup completo
python3.10 -m venv .venv310 && source .venv310/bin/activate
pip install -r requirements.txt && pip install -e .

# Pipeline completa
python -m src.data.download_dataset
python scripts/prepare_data.py
python -m src.training.train_fasttext --config configs/config.yaml
python -m src.training.train_transformer --config configs/config.yaml --fine-tune
python -m src.evaluation.compare_models --config configs/config.yaml

# Test e coverage
pytest tests/ -v --cov=src --cov-report=html

# Servizi (in terminali separati)
mlflow ui --backend-store-uri file:./mlruns --port 5000
python -m src.api.main
streamlit run src/monitoring/dashboard.py
```

---

## 🔧 Troubleshooting

### Problema: Evidently AI non funziona

**Errore**: `TypeError: multiple bases have instance lay-out conflict`

**Soluzione**: Assicurati di usare Python 3.10:
```bash
python --version  # Deve essere 3.10.x
source .venv310/bin/activate
```

### Problema: Modelli non trovati

**Errore**: `FileNotFoundError: models/...`

**Soluzione**: Verifica che i modelli siano stati addestrati:
```bash
ls -lh models/transformer/final_model/
ls -lh models/fasttext/fasttext_model.bin
```

### Problema: Porta già in uso

**Errore**: `Address already in use`

**Soluzione**: Cambia porta:
```bash
# Per API
uvicorn src.api.main:app --port 8001

# Per MLflow
mlflow ui --port 5001

# Per Streamlit
streamlit run src/monitoring/dashboard.py --server.port 8502
```

### Problema: Test falliscono

**Errore**: `ModuleNotFoundError: No module named 'src'`

**Soluzione**: Installa progetto in modalità sviluppo:
```bash
pip install -e .
```

### Problema: Dataset non scaricato

**Errore**: `FileNotFoundError: data/raw/dataset.csv`

**Soluzione**: Esegui download manualmente:
```bash
python -m src.data.download_dataset
```

### Problema: Coverage report non generato

**Errore**: Directory `htmlcov` non trovata

**Soluzione**: Esegui pytest con coverage:
```bash
pytest --cov=src --cov-report=html
```

---

## ⏱️ Tempi Stimati

- **Setup ambiente**: 5-10 minuti
- **Download dati**: 1-2 minuti
- **Preprocessing**: 1 minuto
- **Training FastText**: 2-5 minuti
- **Training Transformer**: 10-30 minuti (CPU)
- **Valutazione**: 2-5 minuti
- **Test e coverage**: 2-3 minuti
- **Generazione report monitoring**: 1-2 minuti

**Totale**: ~30-60 minuti

---

## 📊 Metriche di Successo

### Obiettivi POC

- ✅ Ambiente configurato correttamente
- ✅ Dataset scaricato e preprocessato
- ✅ Entrambi i modelli addestrati
- ✅ Macro-F1 > 0.60 (minimo accettabile)
- ✅ Tutti i test unitari passano
- ✅ Coverage > 80%
- ✅ API funzionante
- ✅ Report Evidently AI generati
- ✅ MLflow tracking attivo
- ✅ Dashboard Streamlit operativa

### Metriche Modelli

- **Transformer**: Macro-F1 > 0.75 (target), Accuracy > 0.75
- **FastText**: Macro-F1 > 0.60 (minimo), Accuracy > 0.60

---

## 📚 Risorse Aggiuntive

### Documentazione

- [Architettura](ARCHITECTURE.md)
- [Modelli](MODELS.md)
- [Deploy](DEPLOYMENT.md)
- [Monitoring](MONITORING.md)
- [Evidently Fix](EVIDENTLY_FIX.md)

### Link Utili

- **MLflow UI**: `http://localhost:5000`
- **API Docs**: `http://localhost:8000/docs`
- **Dashboard**: `http://localhost:8501`

---

## 🎯 Prossimi Passi

Dopo aver completato il POC:

1. **Analizza risultati**: Verifica metriche modelli e coverage
2. **Ottimizza**: Migliora iperparametri se necessario
3. **Deploy**: Prepara deploy su Hugging Face Spaces
4. **Monitoring**: Configura monitoring continuo
5. **Documentazione**: Aggiorna README con risultati

---

**Ultimo Aggiornamento**: 2025-01-05  
**Versione**: 1.0

