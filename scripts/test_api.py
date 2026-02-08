"""
Script per testare API localmente.
"""

import requests
import json
import time

def test_api():
    """Testa endpoint API."""
    base_url = "http://localhost:8000"
    
    print("=== Test API Sentiment Analysis ===\n")
    
    # Test health check
    print("1. Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Health check OK: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ⚠️  API non disponibile. Avvia con: python -m src.api.main")
        return
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        return
    
    # Test lista modelli
    print("\n2. Lista Modelli...")
    try:
        response = requests.get(f"{base_url}/models")
        print(f"   ✅ Modelli disponibili: {response.json()}")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
    
    # Test predizione
    print("\n3. Test Predizione...")
    test_texts = [
        "Questo prodotto è fantastico! Lo consiglio a tutti.",
        "Il servizio è stato ok, niente di speciale.",
        "Terribile esperienza, non lo consiglio affatto.",
    ]
    
    for text in test_texts:
        print(f"\n   Testo: {text}")
        
        # Test Transformer
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"text": text, "model_type": "transformer"},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                print(f"   Transformer: {result['prediction']} (confidence: {result['confidence']:.2f})")
            else:
                print(f"   Transformer: Errore {response.status_code}")
        except Exception as e:
            print(f"   Transformer: Errore - {e}")
        
        # Test FastText
        try:
            response = requests.post(
                f"{base_url}/predict",
                json={"text": text, "model_type": "fasttext"},
                timeout=10
            )
            if response.status_code == 200:
                result = response.json()
                print(f"   FastText: {result['prediction']} (confidence: {result['confidence']:.2f})")
            else:
                print(f"   FastText: Errore {response.status_code}")
        except Exception as e:
            print(f"   FastText: Errore - {e}")
    
    print("\n✅ Test completati!")


if __name__ == "__main__":
    test_api()

