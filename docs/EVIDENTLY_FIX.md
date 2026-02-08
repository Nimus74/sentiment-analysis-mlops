# Risoluzione Problemi Evidently AI - Compatibilità Python 3.13

## Problema Identificato

Evidently AI versione 0.7.18 presenta un problema di compatibilità con Python 3.13.1 e Pydantic 2.10.5.

**Errore**:
```
TypeError: multiple bases have instance lay-out conflict
```

**Causa**: Conflitto tra classi base multiple inheritance in Python 3.13, probabilmente dovuto a incompatibilità tra Pydantic V2 e la struttura interna di Evidently AI.

---

## Soluzioni Disponibili

### Opzione 1: Usare Python 3.11 o 3.10 (Consigliato) ⭐

**Descrizione**: Evidently AI è testato e funzionante con Python 3.11 e 3.10.

**Step Operativi**:

1. **Installare Python 3.11 o 3.10**:
   ```bash
   # macOS (usando Homebrew)
   brew install python@3.11
   
   # Oppure scaricare da python.org
   # https://www.python.org/downloads/
   ```

2. **Creare nuovo ambiente virtuale con Python 3.11**:
   ```bash
   # Rimuovere ambiente esistente (opzionale)
   rm -rf .venv
   
   # Creare nuovo ambiente con Python 3.11
   python3.11 -m venv .venv
   
   # Attivare ambiente
   source .venv/bin/activate  # macOS/Linux
   # oppure
   .venv\Scripts\activate  # Windows
   ```

3. **Installare dipendenze**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Verificare installazione Evidently**:
   ```bash
   python3 -c "import evidently; print('✅ Evidently versione:', evidently.__version__)"
   ```

**Vantaggi**:
- ✅ Soluzione stabile e testata
- ✅ Nessuna modifica al codice necessaria
- ✅ Compatibilità completa con tutte le funzionalità

**Svantaggi**:
- ⚠️ Richiede cambio versione Python
- ⚠️ Potrebbe richiedere reinstallazione dipendenze

---

### Opzione 2: Usare Versione Pre-release di Evidently (Sperimentale)

**Descrizione**: Tentare di installare una versione pre-release o development di Evidently che potrebbe avere fix.

**Step Operativi**:

1. **Disinstallare Evidently corrente**:
   ```bash
   pip uninstall evidently -y
   ```

2. **Installare versione development**:
   ```bash
   pip install git+https://github.com/evidentlyai/evidently.git
   ```

3. **Verificare**:
   ```bash
   python3 -c "import evidently; print('✅ Evidently versione:', evidently.__version__)"
   ```

**Vantaggi**:
- ✅ Mantiene Python 3.13
- ✅ Potrebbe risolvere il problema

**Svantaggi**:
- ⚠️ Versione non stabile
- ⚠️ Potrebbe introdurre altri problemi
- ⚠️ Non garantito che funzioni

---

### Opzione 3: Disabilitare Evidently Temporaneamente (Workaround)

**Descrizione**: Disabilitare le funzionalità di monitoring che usano Evidently fino a quando non sarà disponibile una versione compatibile.

**Step Operativi**:

1. **Modificare `requirements.txt`**:
   ```bash
   # Commentare o rimuovere la riga:
   # evidently>=0.4.0
   ```

2. **Modificare moduli monitoring** per gestire gracefully l'assenza di Evidently:
   ```python
   # In src/monitoring/data_quality.py, data_drift.py, etc.
   try:
       import evidently
   except ImportError:
       evidently = None
       print("⚠️ Evidently AI non disponibile (problema compatibilità Python 3.13)")
   ```

3. **Aggiungere controlli nei moduli**:
   ```python
   if evidently is None:
       raise ImportError(
           "Evidently AI non disponibile. "
           "Per usare il monitoring, installare Python 3.11 o 3.10. "
           "Vedi docs/EVIDENTLY_FIX.md per dettagli."
       )
   ```

**Vantaggi**:
- ✅ Permette di continuare sviluppo senza Evidently
- ✅ Non richiede cambio Python

**Svantaggi**:
- ⚠️ Funzionalità monitoring non disponibili
- ⚠️ Richiede modifiche al codice

---

## Raccomandazione

**Per sviluppo locale**: Usare **Opzione 1** (Python 3.11) per avere tutte le funzionalità disponibili.

**Per produzione**: Se non è possibile cambiare Python, usare **Opzione 3** e implementare monitoring alternativo (es. custom metrics, MLflow tracking avanzato).

---

## Stato Attuale

- **Evidently versione installata**: 0.7.18
- **Python versione**: 3.13.1
- **Pydantic versione**: 2.10.5
- **Compatibilità**: ❌ Non compatibile
- **Fix disponibile**: ⏳ In attesa di aggiornamento Evidently AI

---

## Monitoraggio Issue

- **GitHub Evidently**: https://github.com/evidentlyai/evidently/issues
- **Cerca issue**: "Python 3.13 compatibility" o "multiple bases have instance lay-out conflict"

---

**Ultimo Aggiornamento**: 2025-01-05

