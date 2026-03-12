# Model Monitoring

This project includes experimental monitoring components designed to demonstrate how model performance and data quality can be tracked after deployment.

Monitoring is implemented using **Evidently AI** and visualized through a **Streamlit dashboard**.

The goal is to provide visibility into potential issues that may arise when machine learning models operate in production environments.

---

## Why Monitoring is Important

Machine learning models may degrade over time due to changes in incoming data.

Typical problems include:

- **Data Drift** — the distribution of incoming data changes
- **Prediction Drift** — model predictions change over time
- **Data Quality Issues** — missing or corrupted input data
- **Performance Degradation** — model accuracy decreases

Monitoring helps detect these problems early and supports decisions such as retraining or model updates.

---

## Monitoring Components

The monitoring system in this project includes:

- **Data Quality Analysis**
- **Data Drift Detection**
- **Prediction Drift Monitoring**
- **Model Performance Tracking**

Reports are generated using Evidently AI and can be visualized through an interactive dashboard.

---

## Monitoring Workflow

The monitoring pipeline follows this workflow:

```
Incoming Data
      ↓
Prediction Generation
      ↓
Data & Prediction Logging
      ↓
Evidently Analysis
      ↓
Monitoring Reports
      ↓
Streamlit Dashboard
```

This workflow enables continuous inspection of the system behavior.

---

## Data Quality Monitoring

Data quality monitoring checks whether incoming data respects expected constraints.

Typical checks include:

- missing values
- invalid text inputs
- unexpected data formats
- distribution anomalies

Detecting these issues helps prevent incorrect predictions caused by faulty input data.

---

## Data Drift Detection

Data drift occurs when the statistical distribution of incoming data changes compared to the training dataset.

Examples:

- new vocabulary appearing in user comments
- changes in language style
- shifts in sentiment distribution

Evidently AI computes statistical tests to detect these changes.

---

## Prediction Drift

Prediction drift occurs when the distribution of model predictions changes significantly over time.

For example:

- the model suddenly predicts more negative sentiment
- class probabilities shift unexpectedly
- prediction confidence changes

Monitoring prediction drift can indicate potential model degradation.

---

## Monitoring Dashboard

Monitoring results can be visualized using a Streamlit dashboard.

The dashboard displays:

- data quality reports
- drift analysis
- prediction distribution
- summary metrics

To start the dashboard locally:

```bash
streamlit run src/monitoring/dashboard.py
```

Once started, the dashboard is available at:

```
http://localhost:8501
```

---

## Limitations

The monitoring system included in this project is a **proof-of-concept implementation** intended for demonstration purposes.

Limitations include:

- no automated alerting
- no production logging infrastructure
- reports generated manually rather than continuously

---

## Future Improvements

Potential future extensions include:

- automated monitoring pipelines
- alerting systems for drift detection
- integration with logging platforms
- automated retraining triggers
- production monitoring infrastructure
