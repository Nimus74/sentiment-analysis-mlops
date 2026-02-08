"""
Implementazione modello Transformer per sentiment analysis.
Usa cardiffnlp/twitter-xlm-roberta-base-sentiment (multilingue) da Hugging Face.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from typing import List, Dict, Optional, Tuple
import numpy as np


class TransformerSentimentModel:
    """
    Wrapper per modello Transformer pre-addestrato per sentiment analysis.
    """
    
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",  # Modello multilingue di default
        device: Optional[str] = None,
    ):
        """
        Inizializza il modello Transformer.
        
        Args:
            model_name: Nome modello Hugging Face
            device: Device ('cuda', 'cpu', None per auto)
        """
        self.model_name = model_name
        
        # Determina device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Caricamento modello: {model_name}")
        print(f"Device: {self.device}")
        
        # Carica tokenizer e modello
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()
        
        # Mapping label (il modello usa: LABEL_0, LABEL_1, LABEL_2)
        # Mapping standard: 0=negative, 1=neutral, 2=positive
        self.label_mapping = {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
        }
        
        # Crea pipeline per facilità d'uso
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predice sentiment per un singolo testo.
        
        Args:
            text: Testo da analizzare
        
        Returns:
            Dizionario con predizione e confidence
        """
        result = self.pipeline(text)[0]
        
        # Mappa label
        label = result["label"]
        if label in self.label_mapping:
            label = self.label_mapping[label]
        
        return {
            "label": label,
            "score": float(result["score"]),
            "text": text,
        }
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
        return_probs: bool = False,
    ) -> List[Dict[str, any]]:
        """
        Predice sentiment per un batch di testi.
        
        Args:
            texts: Lista di testi
            batch_size: Dimensione batch
            return_probs: Se True, ritorna probabilità per tutte le classi
        
        Returns:
            Lista di dizionari con predizioni
        """
        results = []
        
        # Processa in batch
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = self.pipeline(batch_texts)
            
            for text, result in zip(batch_texts, batch_results):
                label = result["label"]
                if label in self.label_mapping:
                    label = self.label_mapping[label]
                
                pred = {
                    "label": label,
                    "score": float(result["score"]),
                    "text": text,
                }
                
                if return_probs:
                    # Per ottenere probabilità complete, usa il modello direttamente
                    probs = self._get_probs(text)
                    pred["probabilities"] = probs
                
                results.append(pred)
        
        return results
    
    def _get_probs(self, text: str) -> Dict[str, float]:
        """
        Ottiene probabilità per tutte le classi.
        
        Args:
            text: Testo da analizzare
        
        Returns:
            Dizionario con probabilità per classe
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]
        
        return {
            self.label_mapping[f"LABEL_{i}"]: float(prob)
            for i, prob in enumerate(probs)
        }
    
    def predict_labels(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Predice solo le etichette (senza score) per batch.
        Utile per valutazione con metriche.
        
        Args:
            texts: Lista di testi
            batch_size: Dimensione batch
        
        Returns:
            Array numpy con etichette predette
        """
        predictions = self.predict_batch(texts, batch_size=batch_size)
        
        # Converti label in numeri
        label_to_num = {"negative": 0, "neutral": 1, "positive": 2}
        labels = [label_to_num[pred["label"]] for pred in predictions]
        
        return np.array(labels)
    
    def save(self, save_path: str) -> None:
        """
        Salva modello e tokenizer.
        
        Args:
            save_path: Directory dove salvare
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"Modello salvato: {save_path}")
    
    @classmethod
    def load(cls, load_path: str, device: Optional[str] = None):
        """
        Carica modello salvato.
        
        Args:
            load_path: Directory da cui caricare
            device: Device da usare
        
        Returns:
            Istanza del modello caricato
        """
        import os
        
        # Verifica se è un path locale o un nome modello Hugging Face
        if os.path.exists(load_path) and os.path.isdir(load_path):
            # Path locale: carica modello e tokenizer dalla directory
            print(f"Caricamento modello da directory locale: {load_path}")
            
            # Determina device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Carica tokenizer (se esiste, altrimenti usa quello del modello base)
            tokenizer_path = load_path
            if not os.path.exists(os.path.join(load_path, "tokenizer_config.json")):
                # Se non c'è tokenizer salvato, usa quello del modello base
                print("Tokenizzer non trovato nella directory, uso modello base")
                base_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # Modello multilingue
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # Carica modello
            model = AutoModelForSequenceClassification.from_pretrained(load_path).to(device)
            model.eval()
            
            # Crea istanza
            instance = cls.__new__(cls)
            instance.model_name = load_path
            instance.device = device
            instance.tokenizer = tokenizer
            instance.model = model
            
            # Mapping label
            instance.label_mapping = {
                "LABEL_0": "negative",
                "LABEL_1": "neutral",
                "LABEL_2": "positive",
            }
            
            # Crea pipeline
            instance.pipeline = pipeline(
                "sentiment-analysis",
                model=instance.model,
                tokenizer=instance.tokenizer,
                device=0 if instance.device == "cuda" else -1,
            )
            
            return instance
        else:
            # Nome modello Hugging Face
            return cls(model_name=load_path, device=device)

