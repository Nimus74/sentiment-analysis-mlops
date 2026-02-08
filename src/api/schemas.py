"""
Pydantic schemas per API FastAPI.
Definisce struttura request e response.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class PredictionRequest(BaseModel):
    """Schema per richiesta di predizione."""
    
    text: str = Field(..., description="Testo da analizzare", min_length=1)
    model_type: Literal["transformer", "fasttext"] = Field(
        default="transformer",
        description="Tipo di modello da usare",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Questo prodotto è fantastico!",
                "model_type": "transformer",
            }
        }


class PredictionResponse(BaseModel):
    """Schema per risposta di predizione."""
    
    text: str = Field(..., description="Testo analizzato")
    prediction: str = Field(..., description="Sentiment predetto (negative/neutral/positive)")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    model_used: str = Field(..., description="Modello utilizzato per la predizione")
    probabilities: Optional[dict] = Field(
        None,
        description="Probabilità per tutte le classi (opzionale)",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Questo prodotto è fantastico!",
                "prediction": "positive",
                "confidence": 0.95,
                "model_used": "transformer",
                "probabilities": {
                    "negative": 0.02,
                    "neutral": 0.03,
                    "positive": 0.95,
                },
            }
        }


class HealthResponse(BaseModel):
    """Schema per health check."""
    
    status: str = Field(..., description="Status del servizio")
    models_loaded: dict = Field(..., description="Stato caricamento modelli")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": {
                    "transformer": True,
                    "fasttext": True,
                },
            }
        }


class ModelsResponse(BaseModel):
    """Schema per lista modelli disponibili."""
    
    available_models: list = Field(..., description="Lista modelli disponibili")
    default_model: str = Field(..., description="Modello di default")
    
    class Config:
        json_schema_extra = {
            "example": {
                "available_models": ["transformer", "fasttext"],
                "default_model": "transformer",
            }
        }


class FeedbackRequest(BaseModel):
    """Schema per feedback su predizione."""
    
    text: str = Field(..., description="Testo analizzato")
    prediction: str = Field(..., description="Predizione ricevuta")
    actual_label: Optional[str] = Field(
        None,
        description="Label corretta (se disponibile)",
    )
    model_used: str = Field(..., description="Modello utilizzato")
    feedback_score: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Score feedback utente (1-5)",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Questo prodotto è fantastico!",
                "prediction": "positive",
                "actual_label": "positive",
                "model_used": "transformer",
                "feedback_score": 5,
            }
        }

