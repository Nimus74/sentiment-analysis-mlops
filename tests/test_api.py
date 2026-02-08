"""
Test per API FastAPI.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    """Client di test per API."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data


def test_list_models(client):
    """Test lista modelli disponibili."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "available_models" in data
    assert "default_model" in data
    assert isinstance(data["default_model"], str)  # Deve essere sempre una stringa
    assert isinstance(data["available_models"], list)


def test_predict_endpoint(client):
    """Test endpoint predizione."""
    # Nota: richiede modelli caricati
    response = client.post(
        "/predict",
        json={
            "text": "Questo è un test",
            "model_type": "transformer",
        },
    )
    
    # Potrebbe fallire se modelli non caricati, ma verifica struttura
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "model_used" in data
    elif response.status_code == 503:
        # Modello non disponibile (OK per test senza modelli)
        pass
    else:
        pytest.fail(f"Unexpected status code: {response.status_code}")


def test_predict_invalid_model(client):
    """Test predizione con modello non valido."""
    response = client.post(
        "/predict",
        json={
            "text": "Test",
            "model_type": "invalid_model",
        },
    )
    # Dovrebbe validare e ritornare errore 422 o 503
    assert response.status_code in [422, 503]

