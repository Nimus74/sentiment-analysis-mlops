"""
Test estesi per API FastAPI.
Include test per tutti gli endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    """Client di test per API."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test endpoint root."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data
    assert data["message"] == "Sentiment Analysis API"


def test_health_check_detailed(client):
    """Test health check dettagliato."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], dict)
    assert "transformer" in data["models_loaded"]
    assert "fasttext" in data["models_loaded"]
    assert data["status"] in ["healthy", "degraded"]


def test_predict_positive_text(client):
    """Test predizione con testo positivo."""
    response = client.post(
        "/predict",
        json={
            "text": "Questo prodotto è fantastico! Lo consiglio a tutti.",
            "model_type": "transformer",
        },
    )
    
    if response.status_code == 200:
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert "model_used" in data
        assert data["model_used"] == "transformer"
    elif response.status_code == 503:
        pytest.skip("Modello non disponibile per test")


def test_predict_negative_text(client):
    """Test predizione con testo negativo."""
    response = client.post(
        "/predict",
        json={
            "text": "Terribile esperienza, non lo consiglio affatto.",
            "model_type": "transformer",
        },
    )
    
    if response.status_code == 200:
        data = response.json()
        assert "label" in data
        assert data["label"] in ["negative", "neutral", "positive"]
    elif response.status_code == 503:
        pytest.skip("Modello non disponibile per test")


def test_predict_neutral_text(client):
    """Test predizione con testo neutro."""
    response = client.post(
        "/predict",
        json={
            "text": "Il servizio è stato ok, niente di speciale.",
            "model_type": "transformer",
        },
    )
    
    if response.status_code == 200:
        data = response.json()
        assert "label" in data
        assert data["label"] in ["negative", "neutral", "positive"]
    elif response.status_code == 503:
        pytest.skip("Modello non disponibile per test")


def test_predict_fasttext_model(client):
    """Test predizione con modello FastText."""
    response = client.post(
        "/predict",
        json={
            "text": "Questo è un test",
            "model_type": "fasttext",
        },
    )
    
    if response.status_code == 200:
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert "model_used" in data
        assert data["model_used"] == "fasttext"
    elif response.status_code == 503:
        pytest.skip("Modello non disponibile per test")


def test_predict_empty_text(client):
    """Test predizione con testo vuoto."""
    response = client.post(
        "/predict",
        json={
            "text": "",
            "model_type": "transformer",
        },
    )
    
    # Dovrebbe validare e ritornare errore 422
    assert response.status_code == 422


def test_predict_long_text(client):
    """Test predizione con testo molto lungo."""
    long_text = "Test " * 1000  # Testo molto lungo
    response = client.post(
        "/predict",
        json={
            "text": long_text,
            "model_type": "transformer",
        },
    )
    
    # Dovrebbe gestire correttamente (potrebbe essere troncato)
    assert response.status_code in [200, 422, 503]


def test_feedback_endpoint(client):
    """Test endpoint feedback."""
    response = client.post(
        "/feedback",
        json={
            "text": "Questo è un test",
            "prediction": "positive",
            "actual_label": "positive",
            "model_used": "transformer",
            "feedback_score": 5,
        },
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "message" in data


def test_feedback_endpoint_minimal(client):
    """Test endpoint feedback con dati minimali."""
    response = client.post(
        "/feedback",
        json={
            "text": "Test",
            "prediction": "positive",
            "model_used": "transformer",
        },
    )
    
    assert response.status_code == 200


def test_predict_missing_fields(client):
    """Test predizione con campi mancanti."""
    response = client.post(
        "/predict",
        json={
            "text": "Test",
            # model_type mancante
        },
    )
    
    # Dovrebbe validare e ritornare errore 422 (validazione) o 503 (modello non disponibile)
    assert response.status_code in [422, 503]


def test_predict_invalid_json(client):
    """Test predizione con JSON non valido."""
    response = client.post(
        "/predict",
        json="invalid json",
    )
    
    assert response.status_code == 422

