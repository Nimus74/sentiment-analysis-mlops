# System Architecture

This project implements a sentiment analysis system designed following MLOps principles, with a modular architecture that separates data processing, model training, serving, and monitoring.

The goal is to demonstrate a reproducible and extensible machine learning system lifecycle, from data ingestion to model deployment and monitoring.

---

## High-Level Architecture

The system architecture follows a typical MLOps workflow:

```
Data Sources
      ↓
Data Processing Pipeline
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Artifacts
      ↓
Inference API (FastAPI)
      ↓
Monitoring & Reporting
```

Each component is implemented as an independent module to improve maintainability, scalability, and reproducibility.

---

## Core Components

### Data Pipeline

The data pipeline is responsible for:

- dataset loading
- preprocessing and normalization
- dataset validation
- reproducible train / validation splits

These steps ensure that training experiments remain reproducible and consistent across multiple runs.

---

### Model Training

Two model approaches are implemented in the system.

#### Transformer Model

The primary model used in the project is:

```
cardiffnlp/twitter-roberta-base-sentiment-latest
```

Advantages:

- high performance on short text
- strong results on social media sentiment tasks
- pretrained on large-scale datasets

---

#### FastText Baseline

FastText is included as a baseline model trained within the project.

It provides:

- very fast training
- lightweight inference
- a baseline comparison against the Transformer model

---

### Experiment Tracking

Training experiments can be tracked using MLflow, enabling:

- experiment logging
- parameter tracking
- metric comparison
- artifact storage

This allows reproducible experimentation and easier comparison between models.

---

### Model Serving

The trained models are exposed through a FastAPI inference service.

The API enables:

- real-time sentiment prediction
- dynamic model selection (Transformer or FastText)
- integration with external applications

Example workflow:

```
Client Request
      ↓
FastAPI Service
      ↓
Model Inference
      ↓
Prediction Response
```

---

### Monitoring System

The project includes experimental monitoring components based on Evidently AI.

Monitoring reports include:

- data quality checks
- data drift detection
- prediction drift monitoring
- model performance metrics

These reports can be visualized through a Streamlit dashboard.

---

## Deployment Architecture

The system supports containerized deployment using Docker.

The service can be deployed using:

- Docker
- docker-compose
- local development environments

Containerization ensures consistent runtime environments across development and deployment.

---

## Repository Structure (Simplified)

```
sentiment-analysis-mlops
│
├── src
│   ├── data
│   ├── training
│   ├── models
│   ├── evaluation
│   ├── api
│   └── monitoring
│
├── configs
├── notebooks
├── docs
├── tests
│
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Design Principles

The system is designed following these engineering principles:

- **Modularity** — clear separation between components
- **Reproducibility** — deterministic pipelines and configuration files
- **Experimentation** — ability to compare multiple models
- **Extensibility** — new models and pipelines can be integrated easily
- **Observability** — monitoring and reporting tools are included

---

## Future Improvements

Potential extensions of the system include:

- automated retraining pipelines
- model registry integration
- advanced CI/CD workflows for ML pipelines
- distributed training support
