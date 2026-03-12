# Live Demo

This document describes how to run and test the sentiment analysis system locally.

The repository includes several optional components that allow users to interact with the models and explore monitoring reports.

These components are intended for demonstration and experimentation purposes.

---

# Available Demo Components

The project provides two main demo interfaces:

- **Sentiment Analysis Interface** — interactive inference demo
- **Monitoring Dashboard** — visualization of monitoring reports

Both components are optional and are not required to run the core training pipeline.

---

# Sentiment Analysis Demo

A simple demo interface is available to test the sentiment classification models interactively.

The interface allows users to:

- input custom text
- run sentiment prediction
- compare Transformer and FastText outputs

## Running the Demo

Start the demo application:

```bash
python app.py
```

The application will start a local interface available at:

```
http://127.0.0.1:7860
```

This interface allows users to test the models in real time.

---

# Monitoring Dashboard

The project also includes an experimental monitoring dashboard.

The dashboard visualizes reports generated using Evidently AI.

It can display:

- data quality reports
- data drift analysis
- prediction drift monitoring
- model performance summaries

## Running the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run src/monitoring/dashboard.py
```

The dashboard will be available at:

```
http://localhost:8501
```

---

# Example Workflow

A typical workflow for exploring the project may look like:

1. Train the models
2. Run the inference API
3. Test predictions using the demo interface
4. Generate monitoring reports
5. Explore results in the dashboard

This workflow demonstrates how different components of the system interact within an MLOps-oriented architecture.

---

# Notes

The demo interfaces included in this repository are intended as **proof-of-concept tools**.

They demonstrate how machine learning models can be integrated into user interfaces and monitoring systems but are not designed as production-ready applications.

The primary deliverable of the project remains the machine learning pipeline and the associated experimentation framework.
