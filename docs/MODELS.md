# Confronto Modelli: Transformer vs FastText

## Overview

Il progetto implementa due modelli per sentiment analysis:
1. **Transformer** (cardiffnlp/twitter-roberta-base-sentiment-latest)
2. **FastText** (supervised)

Entrambi sono valutati sullo stesso dataset con le stesse metriche per un confronto equo.

## Modello Transformer

### Caratteristiche

- **Architettura**: RoBERTa base pre-addestrato su Twitter
- **Dimensione**: ~125M parametri
- **Preprocessing**: Tokenizzazione BPE, max length 128
- **Fine-tuning**: Opzionale su dataset italiano

### Vantaggi

- ✅ **Alta accuratezza**: Migliori performance su test complessi
- ✅ **Contesto**: Comprensione contestuale avanzata
- ✅ **Pre-addestrato**: Conoscenza linguistica già incorporata
- ✅ **Fine-tuning**: Adattabile a domini specifici

### Svantaggi

- ❌ **Lentezza**: Inferenza più lenta (~100-200ms per testo)
- ❌ **Risorse**: Richiede GPU per training, più memoria
- ❌ **Costi**: Più costoso computazionalmente

### Performance Attese

- Macro-F1: **0.85-0.90**
- Accuracy: **0.85-0.90**
- Latenza: **100-200ms** (CPU), **10-50ms** (GPU)

## Modello FastText

### Caratteristiche

- **Architettura**: FastText supervised
- **Dimensione**: ~100MB (modello addestrato)
- **Preprocessing**: Formato FastText (__label__<label> <text>)
- **N-grams**: Word n-grams (2) + character n-grams (3-6)

### Vantaggi

- ✅ **Velocità**: Inferenza molto veloce (~1-5ms per testo)
- ✅ **Leggerezza**: Modello piccolo, poca memoria
- ✅ **Retraining**: Facile e veloce da retrainare
- ✅ **Efficienza**: CPU-friendly, no GPU necessaria

### Svantaggi

- ❌ **Accuratezza**: Performance generalmente inferiori a Transformer
- ❌ **Contesto**: Limitata comprensione contestuale
- ❌ **OOV**: Gestione parole fuori vocabolario limitata

### Performance Attese

- Macro-F1: **0.75-0.85**
- Accuracy: **0.75-0.85**
- Latenza: **1-5ms**

## Confronto Diretto

| Metrica | Transformer | FastText | Differenza |
|---------|-------------|----------|------------|
| Macro-F1 | 0.85-0.90 | 0.75-0.85 | +5-10% |
| Accuracy | 0.85-0.90 | 0.75-0.85 | +5-10% |
| Latenza (CPU) | 100-200ms | 1-5ms | 20-200x più veloce |
| Dimensione | ~500MB | ~100MB | 5x più piccolo |
| Training Time | Ore (con GPU) | Minuti | 100x più veloce |
| Retraining | Complesso | Semplice | - |

## Quando Usare Quale Modello

### Usa Transformer quando:

- ✅ **Accuratezza critica**: Hai bisogno della massima precisione
- ✅ **Testi complessi**: Testi con sarcasmo, contesto sottile
- ✅ **Risorse disponibili**: GPU disponibile, latenza non critica
- ✅ **Budget**: Budget computazionale sufficiente

### Usa FastText quando:

- ✅ **Velocità critica**: Inferenza real-time, alta throughput
- ✅ **Risorse limitate**: CPU only, memoria limitata
- ✅ **Retraining frequente**: Aggiornamenti frequenti necessari
- ✅ **Cost-effective**: Costi computazionali bassi

## Trade-off

### Accuratezza vs Velocità

- **Transformer**: Alta accuratezza, bassa velocità
- **FastText**: Media accuratezza, alta velocità

### Costi vs Performance

- **Transformer**: Costi elevati, performance elevate
- **FastText**: Costi bassi, performance buone

### Complessità vs Semplicità

- **Transformer**: Complesso, richiede expertise
- **FastText**: Semplice, facile da mantenere

## Risultati Sperimentali

I risultati effettivi dipendono dal dataset utilizzato. Per un confronto accurato:

1. Esegui `src/evaluation/compare_models.py`
2. Verifica report in `reports/model_comparison/`
3. Controlla metriche su MLflow

## Raccomandazioni

### Per Produzione

- **Default**: Transformer per accuratezza
- **Fallback**: FastText per alta disponibilità
- **A/B Testing**: Testa entrambi e misura metriche business

### Per Sviluppo

- **Prototipo**: FastText per iterazione veloce
- **Finale**: Transformer per performance ottimali

### Per Retraining

- **Frequente**: FastText (settimanale/mensile)
- **Occasionale**: Transformer (trimestrale/annuale)

## Conclusioni

Entrambi i modelli hanno il loro posto nell'ecosistema:

- **Transformer**: Scelta per accuratezza e qualità
- **FastText**: Scelta per velocità e efficienza

Il confronto empirico su dataset reali è essenziale per prendere decisioni informate basate su metriche business specifiche.


# Modelli: Transformer vs FastText

## Obiettivo

Il progetto confronta due approcci per sentiment analysis sullo **stesso dataset** e con una pipeline di preprocessing comune:

1) **Transformer**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (Hugging Face)
2) **FastText**: baseline supervised addestrata nel progetto

Il confronto è pensato per evidenziare trade-off tra accuratezza, semplicità e prestazioni.

## 1) Transformer (Hugging Face)

### Caratteristiche

- Modello RoBERTa pre-addestrato su testi brevi di stile social
- Inferenza tramite pipeline HF
- Adatto a testi “rumorosi” e a pattern linguistici complessi

### Pro

- ✅ Comprensione contestuale più ricca
- ✅ Buone prestazioni “out-of-the-box” su testi tipo social
- ✅ Possibilità di fine-tuning (se richiesto/utile)

### Contro

- ❌ Più costoso computazionalmente rispetto a FastText
- ❌ Latenza maggiore su CPU

## 2) FastText (baseline supervised)

### Caratteristiche

- Modello leggero e veloce
- Addestramento rapido
- Buona scelta come baseline e per scenari CPU-only

### Pro

- ✅ Inferenza molto veloce
- ✅ Poco uso di memoria
- ✅ Training e retraining rapidi

### Contro

- ❌ Comprensione contestuale limitata rispetto ai Transformer
- ❌ Prestazioni dipendenti da feature engineering/preprocessing

## Confronto e risultati

Le metriche (accuracy, macro-F1, ecc.) e le visualizzazioni (confusion matrix) sono riportate nel notebook di consegna:

- `notebooks/DELIVERY_colab_sentiment_analysis.ipynb`

> Nota: i risultati dipendono dal dataset e dalla configurazione di esecuzione; per questo motivo si rimanda al notebook riproducibile.

## Quando usare cosa (indicazioni)

- **Transformer**: quando conta la qualità su testi brevi/rumorosi e la latenza non è l’unico vincolo.
- **FastText**: quando servono semplicità, velocità e risorse limitate, oppure come baseline/fallback.