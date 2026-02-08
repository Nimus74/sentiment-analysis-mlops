# Analisi Performance Modelli - Sentiment Analysis

**Data Analisi**: 2025-01-05  
**Problema**: Performance modelli molto basse rispetto alle aspettative

---

## 📊 Risultati Attuali

### Metriche Complessive
- **Transformer macro-F1**: 0.32 (target: 0.75, gap: -57%)
- **FastText macro-F1**: 0.52 (target: 0.75, gap: -31%)
- **Transformer accuracy**: 0.42 (target: 0.60, gap: -30%)
- **FastText accuracy**: 0.54 (target: 0.60, gap: -10%)

### Performance per Classe - Transformer
| Classe    | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| negative  | 1.00      | 0.01   | **0.01** ❌ | 152     |
| neutral   | 0.36      | 0.96   | **0.53** ✅ | 152     |
| positive  | 0.80      | 0.28   | **0.42** ❌ | 151     |

**Problema Critico**: Transformer ha precision perfetta (1.00) ma recall quasi zero (0.01) per "negative". Questo indica che il modello predice quasi mai "negative", ma quando lo fa è sempre corretto.

### Performance per Classe - FastText
| Classe    | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| negative  | 0.67      | 0.26   | **0.38** ❌ | 152     |
| neutral   | 0.60      | 0.70   | **0.64** ✅ | 152     |
| positive  | 0.46      | 0.66   | **0.54** ✅ | 151     |

**Osservazione**: FastText ha performance più bilanciate ma ancora problemi con "negative".

---

## 🔍 Analisi Dataset

### Distribuzione Classi
- **Training set**: 2122 campioni
  - negative: 707 (33.32%)
  - neutral: 707 (33.32%)
  - positive: 708 (33.36%)
- **Test set**: 455 campioni
  - negative: 152 (33.41%)
  - neutral: 152 (33.41%)
  - positive: 151 (33.19%)

**Conclusione**: ✅ Distribuzione perfettamente bilanciata, nessun problema di squilibrio.

### Qualità Dati
- ✅ Nessun valore nullo
- ✅ Nessun duplicato
- ✅ Lunghezza testi ragionevole (media: 86.4 caratteri, mediana: 90)
- ⚠️ 33.88% dei testi contiene ancora "http" (URL non completamente rimossi?)

### Esempi Dataset
**Negative**:
- "Indiscrezioni Governo Monti: Passera per lo sviluppo pare sia una richiesta part..."
- "http L'errore di Pdl e Pd che non vogliono una base politica per il governo Mont..."

**Neutral**:
- "Controlla questo video Napoli De Magistris chiede al Governo Monti un tavolo su ..."
- "Arco di Constantino @ Roma Italia http http..."

**Positive**:
- "Mario Monti ama la bici. Bene, voglio apprezzarlo per questo motivo e magari sis..."
- "grazie signori............. ma a me piace che posso farci?..."

**Osservazione**: Alcuni esempi "negative" potrebbero essere ambigui o non chiaramente negativi.

---

## 🔧 Analisi Preprocessing

### Confronto RAW vs PROCESSED
- I testi RAW e PROCESSED sono diversi (dataset mescolato o processato)
- Distribuzione classi mantenuta dopo preprocessing (variazioni < 0.1%)
- ⚠️ 33.88% dei testi processati contiene ancora "http"

**Problema Potenziale**: URL non completamente rimossi durante preprocessing.

---

## ⚙️ Analisi Iperparametri

### Transformer - Configurazione Attuale
```yaml
learning_rate: 0.00002  # 2e-5
num_epochs: 3
batch_size: 16
max_length: 128
early_stopping_patience: 2
early_stopping_min_delta: 0.001
```

### Problemi Identificati

1. **Learning Rate Potenzialmente Troppo Basso**
   - 2e-5 è standard per fine-tuning ma potrebbe essere troppo conservativo
   - Con solo 3 epoche, il modello potrebbe non convergere completamente
   - **Raccomandazione**: Provare 3e-5 o 5e-5

2. **Numero Epoche Insufficiente**
   - Solo 3 epoche potrebbero non essere sufficienti per convergenza
   - Con early stopping patience=2, il training potrebbe fermarsi troppo presto
   - **Raccomandazione**: Aumentare a 5-10 epoche

3. **Early Stopping Troppo Aggressivo**
   - Patience=2 con min_delta=0.001 potrebbe fermare il training troppo presto
   - **Raccomandazione**: Aumentare patience a 3-5

4. **Mancanza di Class Weights**
   - Non ci sono class weights per bilanciare le classi (anche se sono bilanciate)
   - Potrebbe aiutare con il problema di recall basso per "negative"
   - **Raccomandazione**: Implementare class weights basati su distribuzione

5. **Mancanza di Warmup Steps**
   - Non ci sono warmup steps nel learning rate scheduler
   - Potrebbe aiutare con la convergenza iniziale
   - **Raccomandazione**: Aggiungere warmup_steps (es. 100-200)

---

## 🎯 Pattern di Errore Identificati

### Transformer
- **Bias verso "neutral"**: 96% recall per "neutral" indica che il modello predice principalmente questa classe
- **Quasi mai predice "negative"**: Recall 0.01 significa che su 152 testi negativi, ne predice correttamente solo ~1-2
- **Precision alta per "negative"**: Quando predice "negative", è sempre corretto (precision 1.00), ma lo fa raramente

### FastText
- **Performance più bilanciate**: Recall più uniforme tra le classi
- **Ancora problemi con "negative"**: F1 0.38 indica difficoltà con questa classe

---

## 💡 Cause Probabili

1. **Modello Base Non Ottimale** ⚠️ **CONFERMATO**
   - `cardiffnlp/twitter-roberta-base-sentiment-latest` è pre-addestrato su inglese
   - **Test eseguito**: Il modello base predice sempre "neutral" per testi negativi italiani
   - **Esempi**:
     - "Terribile esperienza" → neutral (dovrebbe essere negative) ❌
     - "Il prodotto è stato consegnato in ritardo" → neutral (dovrebbe essere negative) ❌
   - **Conclusione**: Il modello base non riconosce correttamente il sentiment negativo in italiano
   - **Impatto**: Il fine-tuning parte da un modello che già ha bias verso "neutral" per testi negativi

2. **Fine-tuning Incompleto**
   - Solo 3 epoche potrebbero non essere sufficienti
   - Learning rate troppo conservativo
   - Early stopping troppo aggressivo

3. **Problemi con Labeling**
   - Alcuni esempi "negative" potrebbero essere ambigui
   - Potrebbe esserci confusione tra "negative" e "neutral"

4. **Preprocessing Incompleto**
   - URL non completamente rimossi (33.88% contiene "http")
   - Potrebbe introdurre rumore nei dati

---

## 🔧 Soluzioni Proposte

### Soluzione 1: Ottimizzare Iperparametri Transformer (Priorità Alta) ⭐

**Modifiche Proposte**:
```yaml
transformer:
  learning_rate: 0.00003  # Aumentato da 2e-5 a 3e-5
  num_epochs: 5  # Aumentato da 3 a 5
  batch_size: 16  # Mantenuto
  max_length: 128  # Mantenuto
  early_stopping_patience: 3  # Aumentato da 2 a 3
  early_stopping_min_delta: 0.0005  # Ridotto da 0.001 a 0.0005
  warmup_steps: 100  # Aggiunto
  weight_decay: 0.01  # Aggiunto per regolarizzazione
```

**Implementazione**: Modificare `configs/config.yaml` e `src/training/train_transformer.py`

---

### Soluzione 2: Implementare Class Weights (Priorità Media)

**Descrizione**: Anche se le classi sono bilanciate, i class weights possono aiutare il modello a dare più importanza alla classe "negative" che ha performance peggiori.

**Implementazione**:
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
```

**Uso nel Trainer**: Passare `class_weights` al modello o usare `weighted_loss` nel Trainer.

---

### Soluzione 3: Migliorare Preprocessing (Priorità Media)

**Problema**: 33.88% dei testi contiene ancora "http"

**Soluzione**: Migliorare la funzione di rimozione URL in `src/data/preprocessing.py`:
```python
def remove_urls(text: str) -> str:
    # Pattern più completo per URL
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    # Rimuovi anche "www." senza http
    text = re.sub(r'www\.[^\s]+', '', text)
    return text.strip()
```

---

### Soluzione 4: Data Augmentation per Classe "Negative" (Priorità Bassa)

**Descrizione**: Aumentare il numero di esempi "negative" attraverso tecniche di data augmentation.

**Tecniche**:
- Back-translation (traduzione italiano→inglese→italiano)
- Sinonimi (sostituzione parole con sinonimi)
- Paraphrasing

**Nota**: Richiede librerie aggiuntive e potrebbe essere complesso.

---

### Soluzione 5: Cambiare Modello Base (Priorità CRITICA) ⭐⭐⭐

**Descrizione**: Il modello base `cardiffnlp/twitter-roberta-base-sentiment-latest` non funziona correttamente su testi italiani negativi.

**Test Eseguito**:
```
✅ "Questo prodotto è fantastico!" → positive (corretto)
✅ "Il servizio è stato ok" → neutral (corretto)
❌ "Terribile esperienza" → neutral (ERRATO, dovrebbe essere negative)
❌ "Il prodotto è stato consegnato in ritardo" → neutral (ERRATO, dovrebbe essere negative)
```

**Problema**: Il modello base predice sempre "neutral" per testi negativi italiani, creando un bias iniziale che il fine-tuning non riesce a correggere.

**Soluzione**: Usare un modello italiano pre-addestrato o un modello multilingue.

**Modelli Alternativi Proposti**:
1. **`dbmdz/bert-base-italian-xxl-cased`** (BERT italiano)
   - Pre-addestrato su corpus italiano
   - Richiede fine-tuning completo (non ha head per sentiment)
   
2. **`nlptown/bert-base-multilingual-uncased-sentiment`** (BERT multilingue)
   - Supporta italiano
   - Ha già head per sentiment (5 classi)
   
3. **`cardiffnlp/twitter-xlm-roberta-base-sentiment`** (XLM-RoBERTa multilingue)
   - Supporta italiano
   - Simile architettura a quello attuale

**Implementazione**: Modificare `configs/config.yaml`:
```yaml
transformer:
  model_name: "nlptown/bert-base-multilingual-uncased-sentiment"  # Cambiato
  # ... altri parametri
```

**Nota**: Potrebbe richiedere adattamento del codice per gestire 5 classi invece di 3 (per modello multilingue sentiment).

---

## 📋 Piano di Azione

### Fase 1: Verifica Modello Base (COMPLETATO) ✅
1. ✅ Testare modello base su esempi italiani
2. ✅ Verificare se le predizioni sono ragionevoli
3. ✅ **RISULTATO**: Modello base non funziona correttamente su testi negativi italiani
4. ⏳ **PROSSIMO STEP**: Cambiare modello base con uno italiano/multilingue

### Fase 2: Cambiare Modello Base (CRITICA) ⭐⭐⭐
1. Scegliere modello alternativo (italiano o multilingue)
2. Modificare `configs/config.yaml` con nuovo `model_name`
3. Adattare codice se necessario (es. gestione 5 classi vs 3)
4. Riaddestrare modello con nuovo modello base
5. Valutare miglioramento

### Fase 3: Ottimizzare Iperparametri (Alta Priorità)
1. Modificare `configs/config.yaml` con nuovi iperparametri
2. Modificare `src/training/train_transformer.py` per supportare warmup_steps e weight_decay
3. Riaddestrare modello con nuovi iperparametri
4. Valutare miglioramento

### Fase 4: Implementare Class Weights (Media Priorità)
1. Aggiungere calcolo class weights in `src/training/train_transformer.py`
2. Passare weights al Trainer o modello
3. Riaddestrare e valutare

### Fase 5: Migliorare Preprocessing (Media Priorità)
1. Migliorare funzione `remove_urls` in `src/data/preprocessing.py`
2. Riprocessare dataset
3. Riaddestrare modelli

### Fase 6: Valutazione Finale
1. Confrontare performance prima/dopo
2. Verificare che macro-F1 > 0.60 (minimo accettabile)
3. Documentare risultati

---

## 🎯 Obiettivi Target

- **Macro-F1**: > 0.60 (minimo accettabile), > 0.75 (target ideale)
- **Accuracy**: > 0.60 (minimo accettabile), > 0.70 (target ideale)
- **F1 per classe**: > 0.50 per tutte le classi
- **Recall "negative"**: > 0.50 (attualmente 0.01)

---

**Ultimo Aggiornamento**: 2025-01-05

