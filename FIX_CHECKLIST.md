# Checklist Risoluzione Problemi - Sentiment Analysis MLOps

Questo documento contiene la checklist per risolvere tutti i problemi identificati durante i test.

**Data Creazione**: 2025-01-05  
**Stato**: In corso

---

## 📋 Problemi Identificati e Priorità

### 🔴 Critici (Bloccanti per produzione)
1. **Test unitari non eseguibili** - Progetto non installato in modalità sviluppo
2. **Performance modelli molto basse** - Macro-F1 Transformer: 0.32 (target: 0.75), FastText: 0.52 (target: 0.75)

### 🟡 Importanti (Richiedono attenzione)
3. **Evidently AI problemi compatibilità** - TypeError: multiple bases have instance lay-out conflict
4. **Gradio mancante nel requirements.txt** - Necessario per deploy Hugging Face Spaces
5. **Logging su stdout invece che su file** - Configurazione non conforme al config.yaml

### 🟢 Minori (Miglioramenti)
6. **Problemi classificazione** - Bias verso "positive" per testi neutri/negativi (già documentato)

---

## ✅ Fix 1: Installare Progetto in Modalità Sviluppo

**Problema**: Test unitari non eseguibili (`ModuleNotFoundError: No module named 'src'`)

**Priorità**: 🔴 Critica

**Causa**: Il progetto non è installato in modalità sviluppo, quindi pytest non trova i moduli `src.*`

**Soluzione**:
1. Verificare che `setup.py` sia corretto ✅ (già verificato)
2. Installare il progetto con `pip install -e .`
3. Verificare che i test unitari funzionino

**Test di Verifica**:
- Eseguire `pytest tests/ -v`
- Verificare che tutti i test unitari passino

**Stato**: ⏳ In attesa di esecuzione

---

## ✅ Fix 2: Aggiungere Gradio al requirements.txt

**Problema**: Gradio utilizzato in `app.py` ma non presente nel `requirements.txt`

**Priorità**: 🟡 Importante

**Causa**: `app.py` usa `import gradio as gr` ma Gradio non è nel requirements.txt

**Soluzione**:
1. Aggiungere `gradio>=4.0.0` al `requirements.txt`
2. Verificare che `app.py` possa essere importato senza errori

**Test di Verifica**:
- Verificare che `import gradio` funzioni
- Verificare che `app.py` possa essere eseguito

**Stato**: ⏳ In attesa di esecuzione

---

## ✅ Fix 3: Configurare Logging su File

**Problema**: L'API logga su stdout invece che su file come configurato in `config.yaml`

**Priorità**: 🟡 Importante

**Causa**: La configurazione di logging in `src/api/main.py` non usa il file configurato

**Soluzione**:
1. Verificare configurazione logging in `config.yaml`
2. Implementare logging su file in `src/api/main.py`
3. Creare directory `logs/` se non esiste
4. Verificare che i log vengano scritti su file

**Test di Verifica**:
- Verificare esistenza file `logs/sentiment_analysis.log`
- Verificare che i log vengano scritti correttamente

**Stato**: ⏳ In attesa di esecuzione

---

## ✅ Fix 4: Risolvere Problemi Compatibilità Evidently AI

**Problema**: `TypeError: multiple bases have instance lay-out conflict` quando si importa Evidently

**Priorità**: 🟡 Importante

**Causa**: Problema di compatibilità tra Evidently AI e Pydantic/Python versione

**Soluzione**:
1. Verificare versione Evidently installata
2. Verificare versione Pydantic
3. Tentare aggiornamento/downgrade Evidently
4. Se necessario, documentare limitazione e suggerire workaround

**Test di Verifica**:
- Verificare che `import evidently` funzioni
- Verificare che i moduli monitoring siano importabili

**Stato**: ⏳ In attesa di esecuzione

**Nota**: Questo potrebbe richiedere downgrade di Python o aggiornamento di Evidently. Se non risolvibile facilmente, documentare la limitazione.

---

## ✅ Fix 5: Investigare e Migliorare Performance Modelli

**Problema**: Performance modelli molto basse rispetto alle aspettative
- Transformer macro-F1: 0.32 (target: 0.75, gap: -57%)
- FastText macro-F1: 0.52 (target: 0.75, gap: -31%)
- Transformer ha problemi gravi con classe "negative" (F1: 0.01)

**Priorità**: 🔴 Critica

**Causa**: Possibili problemi con:
- Dataset di training (squilibrio classi, qualità dati)
- Preprocessing (bias introdotto)
- Iperparametri Transformer (learning rate, epoche, batch size)
- Fine-tuning non ottimale

**Soluzione** (da investigare passo per passo):
1. **Analizzare dataset di training**:
   - Verificare distribuzione classi nel train set
   - Verificare qualità dei dati e labeling
   - Verificare che non ci siano problemi con il parsing del dataset
2. **Verificare preprocessing**:
   - Verificare che il preprocessing non introduca bias
   - Verificare esempi di testi prima/dopo preprocessing
3. **Analizzare confusion matrices**:
   - Identificare pattern di errore
   - Verificare se c'è bias verso una classe specifica
4. **Rivedere iperparametri**:
   - Learning rate: attualmente 2e-5 (potrebbe essere troppo alto/basso)
   - Epoche: attualmente 3 (potrebbe essere insufficiente)
   - Batch size: attualmente 16 (potrebbe essere non ottimale)
5. **Considerare approcci alternativi**:
   - Class weights per bilanciare le classi
   - Data augmentation per classe "negative"
   - Oversampling/undersampling

**Test di Verifica**:
- Eseguire nuovo training con iperparametri ottimizzati
- Verificare che macro-F1 migliori significativamente
- Verificare che tutte le classi abbiano F1 > 0.50

**Stato**: ⏳ In attesa di esecuzione

**Nota**: Questo è il problema più complesso e richiederà più iterazioni. Procederemo passo per passo.

---

## 📝 Note per Esecuzione Fix

### Ordine Consigliato
1. Fix 1 (Installazione progetto) - Prerequisito per altri test
2. Fix 2 (Gradio) - Semplice e veloce
3. Fix 3 (Logging) - Semplice e veloce
4. Fix 4 (Evidently) - Potrebbe richiedere più tempo
5. Fix 5 (Performance modelli) - Complesso, richiede investigazione approfondita

### Criteri di Successo
- ✅ Tutti i test unitari passano dopo Fix 1
- ✅ Gradio installato e funzionante dopo Fix 2
- ✅ Logging su file funzionante dopo Fix 3
- ✅ Evidently importabile dopo Fix 4 (o documentata limitazione)
- ✅ Performance modelli migliorate dopo Fix 5 (macro-F1 > 0.60 minimo)

---

**Ultimo Aggiornamento**: 2025-01-05

