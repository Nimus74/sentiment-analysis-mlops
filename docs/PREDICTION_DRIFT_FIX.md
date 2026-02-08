# Fix Prediction Drift - Evidently AI 0.7.18

**Data**: 2025-01-05  
**Problema**: Prediction Drift richiedeva configurazione aggiuntiva nella nuova API  
**Stato**: ✅ **RISOLTO**

---

## 🔍 Problema Identificato

Nella nuova API di Evidently AI 0.7.18, `ClassificationPreset` richiede una configurazione esplicita tramite `DataDefinition` e `MulticlassClassification`. Senza questa configurazione, si verificava l'errore:

```
ValueError: Cannot use ClassificationPreset without a classification configration
```

---

## ✅ Soluzione Implementata

### Approccio 1: Con Target e Prediction (ClassificationPreset)

Quando sono disponibili sia la colonna `target` (ground truth) che `prediction`, viene utilizzato `ClassificationPreset` con `DataDefinition`:

```python
from evidently import Report, Dataset, DataDefinition, MulticlassClassification
from evidently.presets import ClassificationPreset

# Crea DataDefinition per classificazione multiclasse
data_def = DataDefinition(
    classification=[
        MulticlassClassification(
            target="label",
            prediction_labels="prediction",
        )
    ],
    categorical_columns=["label", "prediction"],
)

# Crea Dataset con DataDefinition
ref_ds = Dataset.from_pandas(reference_data, data_definition=data_def)
cur_ds = Dataset.from_pandas(current_data, data_definition=data_def)

# Crea report con ClassificationPreset
report = Report([ClassificationPreset()])
my_eval = report.run(cur_ds, ref_ds)
```

### Approccio 2: Solo Prediction (DataDriftPreset)

Quando è disponibile solo la colonna `prediction` (senza target), viene utilizzato `DataDriftPreset` per monitorare la distribuzione delle predizioni:

```python
from evidently.presets import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(
    reference_data=reference_data[[prediction_column]], 
    current_data=current_data[[prediction_column]]
)
```

---

## 📝 Modifiche al Codice

### File: `src/monitoring/prediction_drift.py`

**Modifiche Principali**:

1. **Import aggiornati**:
   ```python
   from evidently import Report, Dataset
   from evidently.presets import ClassificationPreset
   from evidently import DataDefinition, MulticlassClassification
   ```

2. **Funzione `create_prediction_drift_report` aggiornata**:
   - Rileva automaticamente la colonna `target` se non specificata
   - Usa `DataDefinition` e `MulticlassClassification` quando target disponibile
   - Fallback a `DataDriftPreset` quando solo prediction disponibile
   - Gestisce correttamente il salvataggio HTML nella nuova API

3. **Calcolo distribuzione drift**:
   - Calcola distribuzioni reference e current
   - Calcola shift percentuale per ogni classe
   - Rileva drift se shift > 15%

---

## ✅ Test Eseguiti

### Test 1: Con Target e Prediction
```python
report, drift_results = create_prediction_drift_report(
    reference_data=df,
    current_data=df.head(20),
    prediction_column="prediction",
    target_column="label",
)
```
**Risultato**: ✅ Report creato con successo (ClassificationPreset)

### Test 2: Solo Prediction
```python
report, drift_results = create_prediction_drift_report(
    reference_data=df[["prediction"]],
    current_data=df[["prediction"]].head(20),
    prediction_column="prediction",
)
```
**Risultato**: ✅ Report creato con successo (DataDriftPreset)

### Test 3: Salvataggio HTML
```python
report, drift_results = create_prediction_drift_report(
    reference_data=df,
    current_data=df.head(20),
    prediction_column="prediction",
    target_column="label",
    output_path="monitoring/reports/prediction_drift_test.html",
)
```
**Risultato**: ✅ Report HTML salvato correttamente (3.5MB)

---

## 📊 Risultati Finali

- ✅ **ClassificationPreset**: Funzionante con DataDefinition
- ✅ **DataDriftPreset fallback**: Funzionante per solo prediction
- ✅ **Report HTML**: Generati correttamente
- ✅ **Calcolo distribuzione**: Corretto
- ✅ **Rilevamento drift**: Funzionante

---

## 🎯 Conclusione

**Prediction Drift è completamente funzionante!**

- ✅ Supporta sia scenario con target che senza target
- ✅ Usa ClassificationPreset quando possibile (più informativo)
- ✅ Fallback a DataDriftPreset quando necessario
- ✅ Compatibile con Python 3.10 e Evidently AI 0.7.18+
- ✅ Report HTML generati correttamente

**Tutti i moduli Evidently AI sono ora completamente operativi!**

---

**Ultimo Aggiornamento**: 2025-01-05

