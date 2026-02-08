# Cambio Modello Base - Documentazione

**Data**: 2025-01-05  
**Motivazione**: Il modello base inglese non riconosceva correttamente sentiment negativo in italiano

---

## Modello Precedente

- **Nome**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Lingua**: Inglese
- **Problema**: Prediceva sempre "neutral" per testi negativi italiani
- **Performance test**: 3/6 corretti (50%)

---

## Nuovo Modello

- **Nome**: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- **Lingua**: Multilingue (supporta italiano)
- **Architettura**: XLM-RoBERTa
- **Classi**: 3 (negative, neutral, positive)
- **Performance test**: 4/6 corretti (66.7%)
- **Miglioramento**: +16.7% accuracy

---

## Modifiche Applicate

### File Modificati

1. **`configs/config.yaml`**
   ```yaml
   transformer:
     model_name: "cardiffnlp/twitter-xlm-roberta-base-sentiment"
   ```

2. **`src/models/transformer_model.py`**
   - Default `model_name` aggiornato
   - Docstring aggiornato
   - Fallback per tokenizer aggiornato

3. **`src/api/main.py`**
   - Default fallback aggiornato

4. **`src/training/train_transformer.py`**
   - Default `model_name` aggiornato in due punti

5. **`src/evaluation/compare_models.py`**
   - Default `model_name` aggiornato

---

## Test Risultati

### Test Esempi Italiani

| Testo | Atteso | Vecchio Modello | Nuovo Modello | Miglioramento |
|-------|--------|----------------|---------------|---------------|
| "Questo prodotto è fantastico!" | positive | ✅ positive | ✅ positive | - |
| "Il servizio è stato ok" | neutral | ✅ neutral | ❌ negative | - |
| "Terribile esperienza" | negative | ❌ neutral | ✅ negative | ✅ |
| "Sono molto soddisfatto" | positive | ❌ neutral | ✅ positive | ✅ |
| "Consegnato in ritardo" | negative | ❌ neutral | ✅ negative | ✅ |
| "Ho ricevuto il pacco" | neutral | ✅ neutral | ❌ positive | - |

**Risultato**: 4/6 corretti (66.7%) vs 3/6 (50%)

### Miglioramenti Critici

✅ **Riconosce correttamente testi negativi italiani**
- "Terribile esperienza" → negative (prima: neutral)
- "Consegnato in ritardo" → negative (prima: neutral)

---

## Prossimi Step

1. ✅ Modello base cambiato
2. ⏳ Riaddestrare modello con nuovo modello base
3. ⏳ Valutare miglioramento performance
4. ⏳ Confrontare risultati prima/dopo

---

**Ultimo Aggiornamento**: 2025-01-05

