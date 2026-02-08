# 📋 Code Review Completa - Sentiment Analysis MLOps

**Data Analisi**: 8 Febbraio 2026  
**Reviewer**: Senior Python Engineer & Code Reviewer  
**Progetto**: Sistema completo end-to-end di analisi del sentiment con architettura MLOps

---

## 1️⃣ VALUTAZIONE GENERALE DEL PROGETTO

### Architettura e Struttura
Il progetto presenta un'architettura ben organizzata con separazione chiara delle responsabilità:
- **Data Pipeline**: Moduli dedicati per download, preprocessing, validazione e split
- **Modelli**: Implementazioni separate per Transformer e FastText con interfacce coerenti
- **MLOps**: Integrazione MLflow per experiment tracking e Evidently AI per monitoring
- **API**: FastAPI per inferenza con supporto multi-backend
- **CI/CD**: GitHub Actions con test automatici

**Punti di Forza**:
- ✅ Struttura modulare e ben organizzata
- ✅ Configurazione centralizzata (YAML)
- ✅ Documentazione presente (README, docs/)
- ✅ Test suite presente
- ✅ Docker support per deployment

**Punti di Attenzione**:
- ⚠️ Discrepanza tra documentazione (menziona Streamlit) e implementazione (usa Gradio)
- ⚠️ Alcuni pattern di preprocessing potrebbero essere migliorati
- ⚠️ Gestione errori non sempre robusta

---

## 2️⃣ FINDINGS PER SEVERITÀ

### 🔴 CRITICAL - Problemi Critici

#### CRIT-1: Preprocessing Applicato Prima dello Split (Rischio Data Leakage)
**File**: `scripts/prepare_data.py`, `src/data/preprocessing.py`  
**Problema**: Il preprocessing viene applicato all'intero dataset prima dello split train/val/test.

**Impatto**: 
- Sebbene il preprocessing sia deterministico (rimozione URL, normalizzazione), applicarlo prima dello split può introdurre inconsistenze se in futuro si aggiungono step che dipendono dalla distribuzione dei dati (es. normalizzazione statistica, vocabolario).

**Spiegazione Tecnica**:
```python
# scripts/prepare_data.py:44-55
df_processed = preprocess_dataframe(df, ...)  # Preprocessing su TUTTO
# Poi split
train_df, val_df, test_df = stratified_split(df_processed, ...)
```

**Raccomandazione**:
- Applicare preprocessing deterministico (rimozione URL, normalizzazione caratteri) dopo lo split è accettabile se documentato
- Per step che dipendono dai dati (es. vocabolario, normalizzazione statistica), applicare solo su training e poi trasformare val/test
- Documentare esplicitamente che il preprocessing è deterministico e non introduce leakage

**Esempio Miglioramento**:
```python
# Split prima
train_df, val_df, test_df = stratified_split(df, ...)

# Preprocessing deterministico (ok applicarlo separatamente)
train_df = preprocess_dataframe(train_df, ...)
val_df = preprocess_dataframe(val_df, ...)
test_df = preprocess_dataframe(test_df, ...)
```

---

#### CRIT-2: Gestione Errori Inconsistente nell'API
**File**: `src/api/main.py`  
**Problema**: Alcuni errori vengono catturati ma non gestiti in modo user-friendly, altri possono causare crash.

**Impatto**: 
- UX degradata per utenti API
- Possibili crash del servizio in produzione
- Difficoltà nel debugging

**Spiegazione Tecnica**:
```python
# src/api/main.py:121-122
except Exception as e:
    logger.error(f"Errore caricamento Transformer: {e}")
    model_cache["transformer"] = None  # OK, ma...
```

Problemi:
- Linea 108: `paths_config` potrebbe non essere definito se config non esiste
- Linea 232-238: Tentativo di ottenere probabilità con fallback silenzioso che può mascherare problemi

**Raccomandazione**:
- Validare configurazione all'avvio con messaggi chiari
- Gestire tutti i casi edge (modello None, testo vuoto, errori di predizione)
- Ritornare errori HTTP appropriati con messaggi informativi

---

#### CRIT-3: Mancanza di Validazione Input nell'API
**File**: `src/api/main.py`, `src/api/schemas.py`  
**Problema**: Validazione Pydantic presente ma non completa per edge cases.

**Impatto**: 
- Possibili errori con input malformati
- Nessuna validazione su lunghezza massima testo
- Nessuna sanitizzazione input

**Spiegazione Tecnica**:
```python
# src/api/schemas.py:13
text: str = Field(..., description="Testo da analizzare", min_length=1)
```

Manca:
- Validazione lunghezza massima (max_length)
- Validazione caratteri speciali pericolosi
- Rate limiting (non implementato)

**Raccomandazione**:
```python
text: str = Field(
    ..., 
    description="Testo da analizzare", 
    min_length=1,
    max_length=512  # Aggiungere
)
```

---

### 🟡 WARNING - Avvisi Importanti

#### WARN-1: Random Seed Non Impostato Globalmente per PyTorch
**File**: `src/training/train_transformer.py`  
**Problema**: Il random seed è impostato solo per sklearn split, ma non per PyTorch/NumPy.

**Impatto**: 
- Riproducibilità non garantita al 100% per training Transformer
- Risultati possono variare tra esecuzioni anche con stesso seed

**Spiegazione Tecnica**:
```python
# src/training/train_transformer.py
# Manca:
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)
```

**Raccomandazione**:
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Chiamare all'inizio di main()
```

---

#### WARN-2: CORS Configurazione Troppo Permissiva
**File**: `src/api/main.py:149`  
**Problema**: CORS permette tutte le origini (`allow_origins=["*"]`).

**Impatto**: 
- Rischio sicurezza in produzione
- Possibili attacchi CSRF

**Spiegazione Tecnica**:
```python
# src/api/main.py:147-153
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Troppo permissivo
    ...
)
```

**Raccomandazione**:
- In produzione, specificare origini esatte
- Usare variabili d'ambiente per configurazione

---

#### WARN-3: Modello FastText - Gestione NumPy 2.x con Monkey Patch
**File**: `src/models/fasttext_model.py:8-13`  
**Problema**: Monkey patch di NumPy per compatibilità FastText.

**Impatto**: 
- Soluzione fragile che può rompersi con aggiornamenti
- Può causare effetti collaterali non previsti

**Spiegazione Tecnica**:
```python
# src/models/fasttext_model.py:8-13
_original_array = np.array
def _patched_array(obj, copy=None, **kwargs):
    if copy is False:
        return np.asarray(obj, **kwargs)
    return _original_array(obj, copy=copy, **kwargs)
np.array = _patched_array
```

**Raccomandazione**:
- Documentare chiaramente il workaround
- Considerare di usare versione compatibile di NumPy (< 2.0) come specificato in requirements.txt
- Monitorare aggiornamenti FastText per fix ufficiale

---

#### WARN-4: Logging di Dati Sensibili Potenziale
**File**: `src/api/main.py:264`  
**Problema**: Feedback viene loggato con `request.dict()` che può contenere dati sensibili.

**Impatto**: 
- Privacy: testi utente nei log
- Conformità GDPR se applicabile

**Spiegazione Tecnica**:
```python
# src/api/main.py:264
logger.info(f"Feedback ricevuto: {request.dict()}")
```

**Raccomandazione**:
- Loggare solo metadata (model_used, prediction, feedback_score)
- Non loggare il testo completo, o almeno sanitizzarlo

---

#### WARN-5: Preprocessing URL Non Completo
**File**: `src/data/preprocessing.py:21-22`  
**Problema**: Pattern regex per URL potrebbe non catturare tutti i casi.

**Impatto**: 
- Secondo docs/PERFORMANCE_ANALYSIS.md, 33.88% dei testi contiene ancora "http"
- Rumore nei dati di training

**Spiegazione Tecnica**:
```python
# src/data/preprocessing.py:21
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
```

**Raccomandazione**:
- Migliorare pattern per catturare anche URL senza protocollo (es. "www.example.com")
- Considerare libreria dedicata (es. `urlextract`)

---

### 🔵 INFO - Osservazioni e Miglioramenti

#### INFO-1: Documentazione vs Implementazione (Streamlit vs Gradio)
**File**: README.md, `.cursor/rules/Code-Review.mdc`, `app.py`  
**Osservazione**: La documentazione menziona Streamlit ma l'implementazione usa Gradio.

**Raccomandazione**: 
- Aggiornare README per riflettere uso di Gradio
- Aggiornare regole code review se non più rilevanti per Streamlit

---

#### INFO-2: Type Hints Parziali
**File**: Vari file  
**Osservazione**: Alcune funzioni hanno type hints, altre no.

**Raccomandazione**: 
- Aggiungere type hints consistenti in tutto il progetto
- Usare `mypy` per validazione statica

---

#### INFO-3: Configurazione Hardcoded in Alcuni Punti
**File**: `src/models/transformer_model.py:24`, `src/models/fasttext_model.py`  
**Osservazione**: Alcuni valori di default sono hardcoded invece di venire da config.

**Raccomandazione**: 
- Centralizzare tutti i default in config.yaml
- Usare dataclass per configurazioni strutturate

---

#### INFO-4: Test Coverage Non Visibile
**File**: `tests/`  
**Osservazione**: Non è chiaro quale sia la copertura dei test.

**Raccomandazione**: 
- Eseguire `pytest --cov=src --cov-report=html` e verificare coverage
- Aggiungere test per edge cases (testo vuoto, molto lungo, caratteri speciali)

---

#### INFO-5: MLflow Model Logging con Fallback Silenzioso
**File**: `src/training/train_transformer.py:305-313`  
**Osservazione**: Se il logging MLflow fallisce, viene solo stampato un warning.

**Raccomandazione**: 
- Decidere se il fallimento del logging è critico o no
- Se critico, sollevare eccezione
- Se non critico, documentare il comportamento

---

### ✅ GOOD - Punti di Forza

#### GOOD-1: Split Stratificato Corretto
**File**: `src/data/split.py`  
**Punto di Forza**: Implementazione corretta di split stratificato con seed fisso.

**Dettagli**:
- ✅ Seed fisso (42) per riproducibilità
- ✅ Stratificazione mantenuta
- ✅ Salvataggio indici per tracciabilità
- ✅ Verifica distribuzione classi

---

#### GOOD-2: Metriche Complete
**File**: `src/evaluation/metrics.py`  
**Punto di Forza**: Implementazione completa di metriche ML con macro-F1 come primaria.

**Dettagli**:
- ✅ Macro-F1, micro-F1, weighted-F1
- ✅ Metriche per classe
- ✅ Confusion matrix
- ✅ Funzione per verificare soglie CI/CD

---

#### GOOD-3: Gestione Modelli con Cache
**File**: `src/api/main.py:79-135`  
**Punto di Forza**: Caricamento modelli con cache e gestione lifecycle.

**Dettagli**:
- ✅ Lifespan context manager per startup/shutdown
- ✅ Cache modelli in memoria
- ✅ Fallback graceful se modello non disponibile
- ✅ Health check endpoint

---

#### GOOD-4: Configurazione Centralizzata
**File**: `configs/config.yaml`  
**Punto di Forza**: Tutta la configurazione in un unico file YAML ben strutturato.

**Dettagli**:
- ✅ Parametri modelli separati
- ✅ Paths configurabili
- ✅ Soglie metriche per CI/CD
- ✅ Configurazione monitoring

---

#### GOOD-5: Docker Support
**File**: `Dockerfile`, `docker-compose.yml`  
**Punto di Forza**: Containerizzazione completa con health check.

**Dettagli**:
- ✅ Dockerfile ottimizzato
- ✅ docker-compose con volumi
- ✅ Health check configurato
- ✅ Restart policy

---

## 3️⃣ TOP 3 RISCHI

### 🥇 RISCHIO #1: Data Leakage Potenziale nel Preprocessing
**Severità**: 🔴 CRITICAL  
**Probabilità**: Media  
**Impatto**: Alto

**Descrizione**: 
Il preprocessing viene applicato prima dello split. Sebbene deterministico, questo pattern può portare a inconsistenze se in futuro si aggiungono step che dipendono dalla distribuzione dei dati.

**Mitigazione**:
1. Documentare esplicitamente che il preprocessing è deterministico
2. Applicare split prima del preprocessing (o almeno documentare l'ordine)
3. Aggiungere test che verificano che il preprocessing non introduce leakage

---

### 🥈 RISCHIO #2: Riproducibilità Training Non Garantita al 100%
**Severità**: 🟡 WARNING  
**Probabilità**: Media  
**Impatto**: Medio

**Descrizione**: 
Random seed non impostato per PyTorch/NumPy nel training Transformer, quindi risultati possono variare tra esecuzioni.

**Mitigazione**:
1. Implementare funzione `set_seed()` completa
2. Chiamarla all'inizio di ogni script di training
3. Aggiungere test di riproducibilità

---

### 🥉 RISCHIO #3: Sicurezza API (CORS Permissivo)
**Severità**: 🟡 WARNING  
**Probabilità**: Bassa (se in produzione)  
**Impatto**: Alto (se in produzione)

**Descrizione**: 
CORS configurato per permettere tutte le origini, rischio sicurezza in produzione.

**Mitigazione**:
1. Configurare CORS con origini specifiche via variabili d'ambiente
2. Aggiungere rate limiting
3. Validazione input più robusta

---

## 4️⃣ QUICK WINS (Alto Impatto, Basso Sforzo)

### ⚡ Quick Win #1: Aggiungere Random Seed Completo
**File**: `src/training/train_transformer.py`  
**Sforzo**: 10 minuti  
**Impatto**: Riproducibilità garantita

```python
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# All'inizio di main()
set_seed(config.get("split", {}).get("random_seed", 42))
```

---

### ⚡ Quick Win #2: Migliorare Validazione Input API
**File**: `src/api/schemas.py`  
**Sforzo**: 5 minuti  
**Impatto**: Robustezza API

```python
text: str = Field(
    ..., 
    min_length=1,
    max_length=512,  # Aggiungere
    description="Testo da analizzare (max 512 caratteri)"
)
```

---

### ⚡ Quick Win #3: Sanitizzare Logging Feedback
**File**: `src/api/main.py:264`  
**Sforzo**: 5 minuti  
**Impatto**: Privacy

```python
# Invece di:
logger.info(f"Feedback ricevuto: {request.dict()}")

# Usare:
logger.info(
    f"Feedback ricevuto: model={request.model_used}, "
    f"prediction={request.prediction}, score={request.feedback_score}"
)
```

---

### ⚡ Quick Win #4: Documentare Preprocessing Order
**File**: `scripts/prepare_data.py`  
**Sforzo**: 5 minuti  
**Impatto**: Chiarezza

Aggiungere commento:
```python
# NOTA: Preprocessing viene applicato prima dello split perché è deterministico
# (rimozione URL, normalizzazione caratteri). Non introduce data leakage.
# Se in futuro si aggiungono step che dipendono dalla distribuzione dei dati,
# applicare solo su training e poi trasformare val/test.
```

---

## 5️⃣ RACCOMANDAZIONI REFACTORING (Medio/Lungo Termine)

### 🔧 Refactor #1: Separare Preprocessing Deterministico da Statistico
**Priorità**: Media  
**Sforzo**: 2-3 giorni

**Descrizione**: 
Creare due categorie di preprocessing:
- **Deterministico**: Può essere applicato prima dello split (rimozione URL, normalizzazione)
- **Statistico**: Deve essere applicato solo su training (vocabolario, normalizzazione statistica)

**Implementazione**:
```python
# src/data/preprocessing.py
def preprocess_deterministic(text: str) -> str:
    """Preprocessing che non dipende dai dati."""
    ...

def preprocess_statistical(texts: List[str], fit_on: List[str]) -> List[str]:
    """Preprocessing che richiede fit su training."""
    ...
```

---

### 🔧 Refactor #2: Configurazione con Dataclass
**Priorità**: Bassa  
**Sforzo**: 1-2 giorni

**Descrizione**: 
Sostituire accesso diretto a dict YAML con dataclass tipizzate.

**Vantaggi**:
- Type safety
- Autocompletamento IDE
- Validazione configurazione all'avvio

**Esempio**:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerConfig:
    model_name: str
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    ...
```

---

### 🔧 Refactor #3: Gestione Errori Centralizzata
**Priorità**: Media  
**Sforzo**: 2 giorni

**Descrizione**: 
Creare exception handler centralizzato per API con logging strutturato.

**Implementazione**:
```python
# src/api/exceptions.py
class SentimentAnalysisException(Exception):
    """Base exception per sentiment analysis."""
    pass

class ModelNotAvailableException(SentimentAnalysisException):
    """Modello non disponibile."""
    pass

# src/api/main.py
@app.exception_handler(SentimentAnalysisException)
async def sentiment_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__}
    )
```

---

### 🔧 Refactor #4: Test Coverage Migliorato
**Priorità**: Alta  
**Sforzo**: 3-4 giorni

**Descrizione**: 
Aumentare coverage test con focus su:
- Edge cases (testo vuoto, molto lungo, caratteri speciali)
- Error handling
- Integrazione end-to-end

**Target**: >80% coverage

---

## 6️⃣ METODOLOGIA ML/NLP - VALUTAZIONE

### ✅ Punti di Forza Metodologici

1. **Split Corretto**: 
   - ✅ Stratificato per mantenere distribuzione classi
   - ✅ Seed fisso per riproducibilità
   - ✅ Indici salvati per tracciabilità

2. **Metriche Appropriate**:
   - ✅ Macro-F1 come metrica principale (corretto per classi bilanciate)
   - ✅ Metriche per classe
   - ✅ Confusion matrix

3. **Early Stopping**:
   - ✅ Implementato per Transformer
   - ✅ Patience configurabile

4. **Validazione Separata**:
   - ✅ Train/Val/Test split corretto
   - ✅ Valutazione su validation set durante training
   - ✅ Test set riservato per valutazione finale

### ⚠️ Punti di Attenzione Metodologici

1. **Preprocessing Order**:
   - ⚠️ Applicato prima dello split (ok se deterministico, ma documentare)

2. **Riproducibilità**:
   - ⚠️ Seed non impostato per PyTorch/NumPy nel training

3. **Feature Engineering**:
   - ⚠️ Nessuna feature engineering avanzata (solo preprocessing base)
   - ⚠️ Potrebbe essere migliorato per FastText

---

## 7️⃣ CONCLUSIONI E PRIORITÀ

### Stato Generale del Progetto
**Voto Complessivo**: 7.5/10

Il progetto è ben strutturato e mostra buone pratiche MLOps. L'architettura è solida e modulare. I principali punti di miglioramento riguardano:
1. Riproducibilità completa del training
2. Gestione errori più robusta
3. Documentazione allineata con implementazione

### Priorità Immediate (Prossima Settimana)
1. 🔴 **CRIT-2**: Migliorare gestione errori API
2. 🟡 **WARN-1**: Aggiungere random seed completo
3. ⚡ **Quick Win #2**: Validazione input API

### Priorità Breve Termine (Prossimo Mese)
1. 🔴 **CRIT-1**: Documentare/refactor preprocessing order
2. 🟡 **WARN-2**: Configurare CORS per produzione
3. 🔧 **Refactor #4**: Migliorare test coverage

### Priorità Medio Termine (Prossimi 3 Mesi)
1. 🔧 **Refactor #1**: Separare preprocessing deterministico/statistico
2. 🔧 **Refactor #3**: Gestione errori centralizzata
3. 🟡 **WARN-3**: Monitorare fix FastText per NumPy 2.x

---

## 📝 NOTE FINALI

Questo progetto dimostra una buona comprensione delle best practices MLOps. Con le correzioni suggerite, sarà production-ready. La struttura modulare facilita manutenzione e estensione futura.

**Raccomandazione Finale**: 
Procedere con le correzioni critiche e i quick wins prima di deploy in produzione. I refactoring possono essere pianificati come miglioramenti incrementali.

---

**Fine Report**
