# Modelli: Transformer vs FastText

## Obiettivo

Il presente documento confronta due approcci per l’analisi del sentiment applicati allo stesso dataset e con una pipeline di preprocessing condivisa:

1. **Transformer**: modello `cardiffnlp/twitter-roberta-base-sentiment-latest` (Hugging Face)
2. **FastText**: modello supervised sviluppato come baseline nel progetto

L’obiettivo è evidenziare le differenze in termini di accuratezza, complessità, prestazioni e applicabilità.

## Modelli

### Transformer

Il modello Transformer utilizzato è basato su RoBERTa pre-addestrato su testi brevi di tipo social. L’inferenza avviene tramite pipeline Hugging Face, che consente una gestione avanzata del contesto linguistico e una buona adattabilità tramite fine-tuning.

**Vantaggi principali:**
- Comprensione contestuale avanzata
- Prestazioni elevate su testi complessi
- Possibilità di adattamento tramite fine-tuning

**Limitazioni:**
- Maggiore complessità computazionale
- Latenza superiore, soprattutto su CPU

### FastText

Il modello FastText è un approccio supervised leggero e veloce, particolarmente indicato per scenari con risorse computazionali limitate. La sua struttura semplice permette un addestramento e un retraining rapidi.

**Vantaggi principali:**
- Inferenza molto veloce e uso contenuto di memoria
- Facilità di addestramento e aggiornamento
- Adatto a deployment in ambienti CPU-only

**Limitazioni:**
- Comprensione contestuale limitata rispetto ai Transformer
- Performance dipendenti dalla qualità del preprocessing e delle feature

## Confronto

I due modelli rappresentano un trade-off tra accuratezza e semplicità operativa. Il Transformer offre una maggiore capacità di interpretare il contesto e testi complessi, mentre FastText privilegia la velocità e l’efficienza computazionale, risultando una scelta adatta per applicazioni con vincoli di risorse o necessità di aggiornamenti frequenti.

La selezione del modello più appropriato dipende quindi dai requisiti specifici del caso d’uso, quali le risorse disponibili, la complessità linguistica dei testi e le esigenze di latenza.

## Risultati

I risultati sperimentali, comprensivi di metriche quali accuracy e macro-F1, oltre a visualizzazioni quali confusion matrix, sono riportati nel notebook di consegna:

- `notebooks/DELIVERY_colab_sentiment_analysis.ipynb`

Si raccomanda di consultare tale risorsa per un’analisi dettagliata e riproducibile delle performance dei modelli.

## Scelte progettuali

- **Transformer**: consigliato quando la qualità dell’analisi è prioritaria e sono disponibili risorse computazionali adeguate, con possibilità di fine-tuning per adattamenti specifici.
- **FastText**: indicato per scenari che richiedono rapidità, semplicità di implementazione e aggiornamenti frequenti, specialmente in ambienti con risorse limitate.
- **Strategia ibrida**: valutare l’adozione di entrambi i modelli in contesti di A/B testing o fallback per bilanciare accuratezza e efficienza.

Entrambi i modelli trovano quindi applicazione complementare nel sistema di analisi del sentiment, in funzione delle esigenze operative e degli obiettivi di business.