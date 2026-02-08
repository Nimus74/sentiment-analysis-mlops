# Data Directory

Questa directory contiene tutti i dati del progetto.

## Struttura

- `raw/`: Dataset originali scaricati da Hugging Face
- `processed/`: Dataset dopo preprocessing
- `splits/`: File con indici train/val/test split
- `reports/`: Report di qualità dati e validazione

## Dataset

Il dataset utilizzato è pubblico e scaricato da Hugging Face. Vedi `src/data/download_dataset.py` per i dettagli.

## Note

I file di dati non sono inclusi nel repository per questioni di dimensione. Usa lo script di download per ottenere i dati.

