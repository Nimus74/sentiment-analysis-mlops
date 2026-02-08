FROM python:3.10-slim

# Imposta working directory
WORKDIR /app

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice applicazione
COPY . .

# Crea directory necessarie
RUN mkdir -p data/raw data/processed data/splits models/transformer models/fasttext \
    monitoring/reports logs mlruns

# Esponi porta API
EXPOSE 8000

# Variabili d'ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Comando di avvio
CMD ["python", "-m", "src.api.main"]

