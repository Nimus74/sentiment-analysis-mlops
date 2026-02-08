"""
API FastAPI per inferenza sentiment analysis.
Supporta selezione backend (Transformer o FastText).
"""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager
import yaml
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelsResponse,
    FeedbackRequest,
)
from src.models.transformer_model import TransformerSentimentModel
from src.models.fasttext_model import FastTextSentimentModel

# Setup logging da config.yaml
def setup_logging():
    """Configura logging da config.yaml."""
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging_config = config.get("logging", {})
    else:
        logging_config = {}
    
    log_level = logging_config.get("level", "INFO")
    log_format = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = logging_config.get("file", "logs/sentiment_analysis.log")
    
    # Crea directory logs se non esiste
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configura logging con file handler
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Mantiene anche output su stdout
        ]
    )

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Cache modelli
model_cache = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestisce lifecycle dell'applicazione (startup/shutdown)."""
    # Startup: carica modelli
    logger.info("Caricamento modelli...")
    load_models()
    logger.info("Modelli caricati")
    
    yield
    
    # Shutdown: cleanup
    logger.info("Shutdown applicazione")


def load_models():
    """Carica modelli in cache."""
    # Carica configurazione
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        logger.warning("Config file non trovato, usando default")
        transformer_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # Modello multilingue
        fasttext_model_path = "models/fasttext/fasttext_model.bin"
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        transformer_config = config.get("transformer", {})
        fasttext_config = config.get("fasttext", {})
        paths_config = config.get("paths", {})
        
        transformer_model_name = transformer_config.get(
            "model_name",
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",  # Modello multilingue di default
        )
        fasttext_model_path = os.path.join(
            paths_config.get("models_fasttext", "models/fasttext"),
            "fasttext_model.bin",
        )
    
    # Carica Transformer
    try:
        # Prova a caricare modello fine-tuned se esiste
        transformer_path = os.path.join(
            paths_config.get("models_transformer", "models/transformer"),
            "final_model",
        )
        if os.path.exists(transformer_path):
            logger.info(f"Caricamento Transformer da: {transformer_path}")
            model_cache["transformer"] = TransformerSentimentModel.load(transformer_path)
        else:
            logger.info(f"Caricamento Transformer pre-addestrato: {transformer_model_name}")
            model_cache["transformer"] = TransformerSentimentModel(
                model_name=transformer_model_name
            )
        logger.info("✅ Transformer caricato")
    except Exception as e:
        logger.error(f"Errore caricamento Transformer: {e}")
        model_cache["transformer"] = None
    
    # Carica FastText
    try:
        if os.path.exists(fasttext_model_path):
            logger.info(f"Caricamento FastText da: {fasttext_model_path}")
            model_cache["fasttext"] = FastTextSentimentModel.load(fasttext_model_path)
            logger.info("✅ FastText caricato")
        else:
            logger.warning(f"Modello FastText non trovato: {fasttext_model_path}")
            model_cache["fasttext"] = None
    except Exception as e:
        logger.error(f"Errore caricamento FastText: {e}")
        model_cache["fasttext"] = None


# Crea app FastAPI
app = FastAPI(
    title="Sentiment Analysis API",
    description="API per analisi sentiment con Transformer e FastText",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione, specificare origini
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Endpoint root."""
    return {
        "message": "Sentiment Analysis API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    models_loaded = {
        "transformer": model_cache.get("transformer") is not None,
        "fasttext": model_cache.get("fasttext") is not None,
    }
    
    status = "healthy" if all(models_loaded.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
    )


@app.get("/models", response_model=ModelsResponse, tags=["General"])
async def list_models():
    """Lista modelli disponibili."""
    available = []
    if model_cache.get("transformer"):
        available.append("transformer")
    if model_cache.get("fasttext"):
        available.append("fasttext")
    
    # Ensure default_model is always a string (never None)
    if "transformer" in available:
        default_model_name = "transformer"
    elif available:
        default_model_name = available[0]
    else:
        default_model_name = "none"  # Nessun modello disponibile
    
    return ModelsResponse(
        available_models=available,
        default_model=default_model_name,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predice sentiment di un testo.
    
    Args:
        request: Richiesta con testo e tipo modello
    
    Returns:
        Predizione con sentiment e confidence
    """
    model_type = request.model_type
    
    # Verifica modello disponibile
    model = model_cache.get(model_type)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modello {model_type} non disponibile",
        )
    
    try:
        # Predizione
        result = model.predict(request.text)
        
        # Ottieni probabilità se disponibili
        probabilities = None
        if hasattr(model, "predict_batch"):
            try:
                probs_result = model.predict_batch([request.text], return_probs=True)
                if probs_result and "probabilities" in probs_result[0]:
                    probabilities = probs_result[0]["probabilities"]
            except Exception:
                pass
        
        return PredictionResponse(
            text=request.text,
            prediction=result["label"],
            confidence=result["score"],
            model_used=model_type,
            probabilities=probabilities,
        )
    
    except Exception as e:
        logger.error(f"Errore predizione: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Errore durante predizione: {str(e)}",
        )


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Raccoglie feedback su predizioni.
    Utile per monitoring e retraining.
    """
    # Qui potresti salvare in database o file
    # Per ora solo logging
    logger.info(f"Feedback ricevuto: {request.dict()}")
    
    # Salva feedback in file (in produzione usare database)
    feedback_file = "data/feedback.jsonl"
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    
    import json
    with open(feedback_file, "a") as f:
        f.write(json.dumps(request.dict()) + "\n")
    
    return {"status": "feedback_received", "message": "Grazie per il feedback!"}


def main():
    """Entry point per avvio server."""
    # Carica configurazione API
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        api_config = config.get("api", {})
    else:
        api_config = {}
    
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

