# Deploy & Execution Guide (POC)

Questo documento descrive **come eseguire il progetto in locale** e chiarisce le **opzioni di deploy opzionali**.
Il progetto è pensato principalmente per essere valutato tramite **Notebook Google Colab**.

---

## 1. Esecuzione Locale (API)

### Prerequisiti
- Python 3.10+
- (Opzionale) Docker

### Setup ambiente

```bash
git clone https://github.com/Nimus74/sentiment-analysis-mlops.git
cd sentiment-analysis-mlops

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Avvio API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

API disponibile su: `http://localhost:8000`

### Test rapido

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Questo prodotto è fantastico!","model_type":"transformer"}'
```

---

## 2. Docker (opzionale – POC)

Per eseguire il progetto tramite Docker:

```bash
docker-compose up --build
```

Questo approccio è fornito come **proof-of-concept** e non come setup production-ready.

---

## 3. Hugging Face Spaces (opzionale)

Il repository include un file `app.py` utilizzabile per una demo su **Hugging Face Spaces**.

Questo step è:
- completamente **facoltativo**
- **non richiesto** per la consegna
- fornito come estensione dimostrativa

Indicazioni generali:
- creare uno Space su Hugging Face
- scegliere uno SDK coerente con `app.py`
- assicurarsi che `requirements.txt` includa tutte le dipendenze

---

## 4. Nota per la Consegna

La **consegna ufficiale del progetto** è rappresentata dal **Notebook Google Colab**, linkato nel `README.md`.

Il deploy è incluso solo per dimostrare:
- capacità di esposizione API
- organizzazione MLOps del progetto
- possibilità di estensione verso ambienti reali