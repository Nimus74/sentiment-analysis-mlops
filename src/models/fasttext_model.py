"""
Implementazione modello FastText supervised per sentiment analysis.
Baseline per confronto con Transformer.
"""

import numpy as np
# Monkey patch per compatibilità NumPy 2.x con fasttext
_original_array = np.array
def _patched_array(obj, **kwargs):
    # numpy.array() accetta parametri che np.asarray() non supporta (es. subok, copy).
    # Alcune librerie (scipy/evidently) passano subok/copy: li rimuoviamo per evitare crash.
    kwargs.pop("subok", None)
    kwargs.pop("copy", None)
    return np.asarray(obj, **kwargs)
np.array = _patched_array

import fasttext
from typing import List, Dict, Optional
from pathlib import Path
import tempfile
import os
import warnings

# Suppress NumPy 2.x warnings per fasttext
warnings.filterwarnings('ignore', category=RuntimeWarning)


class FastTextSentimentModel:
    """
    Wrapper per modello FastText supervised per sentiment analysis.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[fasttext.FastText._FastText] = None,
    ):
        """
        Inizializza il modello FastText.
        
        Args:
            model_path: Path al modello salvato (.bin)
            model: Modello FastText già caricato (opzionale)
        """
        if model is not None:
            self.model = model
        elif model_path and os.path.exists(model_path):
            print(f"Caricamento modello FastText: {model_path}")
            self.model = fasttext.load_model(model_path)
        else:
            raise ValueError(
                "Devi fornire model_path o model. Usa train() per addestrare."
            )
        
        # Mapping label (FastText usa formato __label__<label>)
        self.label_mapping = {
            "__label__negative": "negative",
            "__label__neutral": "neutral",
            "__label__positive": "positive",
        }
        
        # Ordine label per probabilità
        self.labels = ["negative", "neutral", "positive"]
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predice sentiment per un singolo testo.
        
        Args:
            text: Testo da analizzare
        
        Returns:
            Dizionario con predizione e confidence
        """
        # FastText predice (label, score)
        # Gestione compatibilità NumPy 2.x - fasttext restituisce (labels, scores)
        result = self.model.predict(text, k=1)
        
        # FastText restituisce sempre una tupla (labels, scores)
        if isinstance(result, tuple) and len(result) == 2:
            labels, scores = result
            # Estrai primo elemento
            label_raw = labels[0] if isinstance(labels, (list, tuple, np.ndarray)) else labels
            # Converti score in float, gestendo array numpy
            if isinstance(scores, np.ndarray):
                score = float(scores[0])
            elif isinstance(scores, (list, tuple)):
                score = float(scores[0])
            else:
                score = float(scores)
        else:
            # Fallback per formati diversi
            label_raw, score = result[0] if isinstance(result, (list, tuple)) else result
            score = float(score)
        
        # Rimuovi prefisso __label__
        label = label_raw.replace("__label__", "")
        
        return {
            "label": label,
            "score": float(score),
            "text": text,
        }
    
    def predict_batch(
        self,
        texts: List[str],
        return_probs: bool = False,
    ) -> List[Dict[str, any]]:
        """
        Predice sentiment per un batch di testi.
        
        Args:
            texts: Lista di testi
            return_probs: Se True, ritorna probabilità per tutte le classi
        
        Returns:
            Lista di dizionari con predizioni
        """
        results = []
        
        for text in texts:
            if return_probs:
                # Ottieni probabilità per tutte le classi
                predictions = self.model.predict(text, k=3)
                labels_raw = [p[0] for p in predictions]
                scores = [float(p[1]) for p in predictions]
                
                # Crea dizionario probabilità
                probs = {}
                for label_raw, score in zip(labels_raw, scores):
                    label = label_raw.replace("__label__", "")
                    probs[label] = score
                
                # Label principale
                main_label = labels_raw[0].replace("__label__", "")
                
                pred = {
                    "label": main_label,
                    "score": float(scores[0]),
                    "probabilities": probs,
                    "text": text,
                }
            else:
                pred = self.predict(text)
            
            results.append(pred)
        
        return results
    
    def predict_labels(self, texts: List[str]) -> np.ndarray:
        """
        Predice solo le etichette (senza score) per batch.
        Utile per valutazione con metriche.
        
        Args:
            texts: Lista di testi
        
        Returns:
            Array numpy con etichette predette
        """
        predictions = self.predict_batch(texts)
        
        # Converti label in numeri
        label_to_num = {"negative": 0, "neutral": 1, "positive": 2}
        labels = [label_to_num[pred["label"]] for pred in predictions]
        
        return np.array(labels)
    
    @classmethod
    def train(
        cls,
        train_file: str,
        output_path: str,
        lr: float = 0.1,
        epoch: int = 25,
        wordNgrams: int = 2,
        dim: int = 100,
        minCount: int = 1,
        minn: int = 3,
        maxn: int = 6,
        bucket: int = 2000000,
    ):
        """
        Addestra un nuovo modello FastText.
        
        Args:
            train_file: Path al file di training (formato FastText)
            output_path: Path dove salvare il modello (.bin)
            lr: Learning rate
            epoch: Numero di epoche
            wordNgrams: N-grammi di parole
            dim: Dimensione vettori
            minCount: Frequenza minima parola
            minn: Lunghezza minima caratteri n-gram
            maxn: Lunghezza massima caratteri n-gram
            bucket: Numero di bucket
        
        Returns:
            Istanza del modello addestrato
        """
        print(f"Training FastText su: {train_file}")
        
        # Addestra modello
        model = fasttext.train_supervised(
            train_file,
            lr=lr,
            epoch=epoch,
            wordNgrams=wordNgrams,
            dim=dim,
            minCount=minCount,
            minn=minn,
            maxn=maxn,
            bucket=bucket,
        )
        
        # Salva modello
        model.save_model(output_path)
        print(f"Modello salvato: {output_path}")
        
        # Valuta su training set
        result = model.test(train_file)
        print(f"Training accuracy: {result[1]:.4f}")
        print(f"Training samples: {result[0]}")
        
        return cls(model=model)
    
    def save(self, save_path: str) -> None:
        """
        Salva modello.
        
        Args:
            save_path: Path dove salvare (.bin)
        """
        self.model.save_model(save_path)
        print(f"Modello salvato: {save_path}")
    
    @classmethod
    def load(cls, load_path: str):
        """
        Carica modello salvato.
        
        Args:
            load_path: Path al modello (.bin)
        
        Returns:
            Istanza del modello caricato
        """
        return cls(model_path=load_path)

