# Esecuzione del progetto (locale) – guida breve (opzionale)

> **Consegna principale**: notebook Google Colab `notebooks/DELIVERY_colab_sentiment_analysis.ipynb` (linkato nel README).
> 
> Questa guida descrive un’esecuzione **locale opzionale** per chi volesse riprodurre pipeline, test e servizi.

---

## 1) Setup

```bash
git clone https://github.com/Nimus74/sentiment-analysis-mlops.git
cd sentiment-analysis-mlops

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Nota**: alcune dipendenze (es. `evidently`, `torch/transformers`) possono richiedere una versione di Python compatibile con la piattaforma. Se riscontri errori di installazione, usa la versione indicata in `requirements.txt`/documentazione del corso oppure esegui direttamente il notebook Colab.

---

## 2) Dataset + preprocessing + split

```bash
python -m src.data.download_dataset
python scripts/prepare_data.py
```

**Output atteso (locale)**:
- file di split in `data/splits/` (train/val/test)
- dati preprocessati in `data/processed/`

---

## 3) Training (opzionale)

```bash
python -m src.training.train_fasttext --config configs/config.yaml
# (opzionale) training/fine-tuning Transformer se previsto dalla configurazione
python -m src.training.train_transformer --config configs/config.yaml
```

---

## 4) Valutazione / confronto modelli

```bash
python -m src.evaluation.compare_models --config configs/config.yaml
```

**Output atteso**:
- report di confronto in `reports/model_comparison/`

---

## 5) Test

```bash
pytest -v
```

---

## 6) API (FastAPI)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

---

## 7) Monitoring (POC)

```bash
python -m src.monitoring.data_quality
python -m src.monitoring.data_drift
python -m src.monitoring.prediction_drift

streamlit run src/monitoring/dashboard.py
```

**Nota**: i componenti di monitoring sono dimostrativi (proof-of-concept) e pensati per mostrare l’approccio MLOps; non rappresentano un’implementazione production-ready.