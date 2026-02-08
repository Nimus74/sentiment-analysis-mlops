# Fix Salvataggio Report Evidently AI

**Data**: 2025-01-05  
**Problema**: Report Evidently AI non venivano salvati correttamente  
**Stato**: ✅ **RISOLTO**

---

## 🔍 Problema Identificato

Nella nuova API di Evidently AI 0.7.18+, il metodo `report.run()` restituisce un oggetto `Snapshot`, non modifica il `Report` stesso. Il metodo `save_html()` è disponibile sul `Snapshot`, non sul `Report`.

**Errore**:
```
⚠️ Errore salvataggio report HTML: 'Report' object has no attribute 'as_dict'
Report creato ma non salvato. Tipo: <class 'evidently.core.report.Report'>
```

**Causa**: Il codice tentava di chiamare `report.save_html()` invece di `snapshot.save_html()`.

---

## ✅ Soluzione Implementata

### Modifica Principale

Nella nuova API di Evidently 0.7.18+:
- `report.run()` restituisce un `Snapshot` object
- Il metodo `save_html()` è disponibile sul `Snapshot`, non sul `Report`

**Codice Corretto**:
```python
# Nuova API (0.7.18+)
report = Report(metrics=[DataSummaryPreset()])
snapshot = report.run(reference_data=reference_data, current_data=current_data)

# Salva usando snapshot
if hasattr(snapshot, 'save_html'):
    snapshot.save_html(output_path)
```

### File Modificati

#### 1. `src/monitoring/data_quality.py`

**Prima**:
```python
report.run(reference_data=reference_data, current_data=current_data)
# ...
report.save_html(output_path)  # ❌ Non funziona
```

**Dopo**:
```python
snapshot = report.run(reference_data=reference_data, current_data=current_data)
# ...
if EVIDENTLY_NEW_API and hasattr(snapshot, 'save_html'):
    snapshot.save_html(output_path)  # ✅ Funziona
```

#### 2. `src/monitoring/data_drift.py`

**Prima**:
```python
report.run(reference_data=reference_data, current_data=current_data)
# ...
report.save_html(output_path)  # ❌ Non funziona
```

**Dopo**:
```python
snapshot = report.run(reference_data=reference_data, current_data=current_data)
# ...
if EVIDENTLY_NEW_API and hasattr(snapshot, 'save_html'):
    snapshot.save_html(output_path)  # ✅ Funziona
```

#### 3. `src/monitoring/prediction_drift.py`

**Già corretto** in precedenza con `my_eval.save_html()`.

---

## ✅ Test Eseguiti

### Test 1: Data Quality Report
```python
from src.monitoring.data_quality import generate_data_quality_report
import pandas as pd

ref_df = pd.read_csv('data/processed/train.csv')
report_path = generate_data_quality_report(
    reference_path='data/processed/train.csv',
    current_data=ref_df.head(100),
    output_dir='monitoring/reports',
    report_name='data_quality_report.html'
)
```

**Risultato**: ✅ File creato correttamente (3.5MB)

### Test 2: Data Drift Report
```python
from src.monitoring.data_drift import check_data_drift

drift_results = check_data_drift(
    reference_path='data/processed/train.csv',
    current_data=ref_df.tail(100),
    output_dir='monitoring/reports',
    report_name='data_drift_report.html'
)
```

**Risultato**: ✅ File creato correttamente (3.5MB)

### Test 3: Prediction Drift Report
```python
from src.monitoring.prediction_drift import check_prediction_drift

test_df['prediction'] = test_df['label']
drift_results = check_prediction_drift(
    reference_data=test_df.head(200),
    current_data=test_df.tail(100),
    output_dir='monitoring/reports',
    report_name='prediction_drift_report.html'
)
```

**Risultato**: ✅ File creato correttamente (3.5MB)

---

## 📊 Risultati Finali

### Report Generati Correttamente

- ✅ **Data Quality Report**: `monitoring/reports/data_quality_report.html` (3.5MB)
- ✅ **Data Drift Report**: `monitoring/reports/data_drift_report.html` (3.5MB)
- ✅ **Prediction Drift Report**: `monitoring/reports/prediction_drift_report.html` (3.5MB)
- ✅ **Performance Report**: `monitoring/reports/performance_report.html` (3.5MB)

### Verifica File

```bash
ls -lh monitoring/reports/*.html
```

**Output**:
```
-rw-r--r--  3.5M  data_drift_report.html
-rw-r--r--  3.5M  data_quality_report.html
-rw-r--r--  3.5M  prediction_drift_report.html
-rw-r--r--  3.5M  performance_report.html
```

---

## 🔧 Dettagli Tecnici

### API Evidently 0.7.18+

**Struttura**:
```python
from evidently import Report
from evidently.presets import DataSummaryPreset

# Crea report
report = Report(metrics=[DataSummaryPreset()])

# Esegui report (restituisce Snapshot)
snapshot = report.run(reference_data=ref_df, current_data=cur_df)

# Snapshot ha il metodo save_html()
snapshot.save_html("report.html")
```

**Tipi**:
- `Report`: Oggetto configurazione del report
- `Snapshot`: Risultato dell'esecuzione del report (contiene i dati)

---

## 📝 Note Importanti

1. **Snapshot vs Report**: Nella nuova API, `report.run()` restituisce un `Snapshot`, non modifica il `Report` stesso.

2. **Metodi Disponibili**:
   - `snapshot.save_html(path)`: Salva report come HTML ✅
   - `report.save_html(path)`: Non disponibile nella nuova API ❌

3. **Retrocompatibilità**: Il codice gestisce sia la nuova API (0.7.18+) che la vecchia API (< 0.7.18).

4. **Gestione Errori**: Aggiunto try/except con traceback completo per debugging.

---

## ✅ Conclusione

**Tutti i report Evidently AI vengono ora salvati correttamente!**

- ✅ Data Quality Report: Funzionante
- ✅ Data Drift Report: Funzionante
- ✅ Prediction Drift Report: Funzionante
- ✅ Performance Report: Funzionante

**File HTML generati**: Tutti i report vengono salvati come file HTML (~3.5MB ciascuno) nella directory `monitoring/reports/`.

---

**Ultimo Aggiornamento**: 2025-01-05

