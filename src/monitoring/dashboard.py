"""
Dashboard monitoring centralizzata con Streamlit.
Aggrega tutti i report Evidently AI.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime
import os

from src.monitoring.data_quality import create_data_quality_report
from src.monitoring.data_drift import create_data_drift_report
from src.monitoring.prediction_drift import create_prediction_drift_report


def load_latest_reports(reports_dir: str = "monitoring/reports"):
    """Carica ultimi report generati."""
    reports = {}
    
    report_files = {
        "data_quality": "data_quality_report.html",
        "data_drift": "data_drift_report.html",
        "prediction_drift": "prediction_drift_report.html",
        "performance": "performance_report.html",
    }
    
    for report_type, filename in report_files.items():
        report_path = os.path.join(reports_dir, filename)
        if os.path.exists(report_path):
            reports[report_type] = report_path
    
    return reports


def main():
    """Funzione principale dashboard Streamlit."""
    st.set_page_config(
        page_title="Sentiment Analysis Monitoring",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("📊 Sentiment Analysis Monitoring Dashboard")
    st.markdown("Dashboard centralizzata per monitoring sistema MLOps")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Seleziona pagina",
        ["Overview", "Data Quality", "Data Drift", "Prediction Drift", "Performance"],
    )
    
    reports_dir = st.sidebar.text_input(
        "Reports Directory",
        value="monitoring/reports",
    )
    
    # Carica report disponibili
    reports = load_latest_reports(reports_dir)
    
    if page == "Overview":
        st.header("Overview Monitoring")
        
        # Statistiche generali
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Report Disponibili", len(reports))
        
        with col2:
            st.metric("Data Quality", "✓" if "data_quality" in reports else "✗")
        
        with col3:
            st.metric("Data Drift", "✓" if "data_drift" in reports else "✗")
        
        with col4:
            st.metric("Prediction Drift", "✓" if "prediction_drift" in reports else "✗")
        
        # Lista report
        st.subheader("Report Disponibili")
        for report_type, report_path in reports.items():
            st.write(f"- **{report_type}**: {report_path}")
            if st.button(f"Visualizza {report_type}", key=report_type):
                st.components.v1.html(
                    open(report_path, "r").read(),
                    height=600,
                    scrolling=True,
                )
    
    elif page == "Data Quality":
        st.header("Data Quality Monitoring")
        
        if "data_quality" in reports:
            st.components.v1.html(
                open(reports["data_quality"], "r").read(),
                height=800,
                scrolling=True,
            )
        else:
            st.warning("Nessun report data quality disponibile")
    
    elif page == "Data Drift":
        st.header("Data Drift Detection")
        
        if "data_drift" in reports:
            st.components.v1.html(
                open(reports["data_drift"], "r").read(),
                height=800,
                scrolling=True,
            )
        else:
            st.warning("Nessun report data drift disponibile")
    
    elif page == "Prediction Drift":
        st.header("Prediction Drift Monitoring")
        
        if "prediction_drift" in reports:
            st.components.v1.html(
                open(reports["prediction_drift"], "r").read(),
                height=800,
                scrolling=True,
            )
        else:
            st.warning("Nessun report prediction drift disponibile")
    
    elif page == "Performance":
        st.header("Model Performance Monitoring")
        
        if "performance" in reports:
            st.components.v1.html(
                open(reports["performance"], "r").read(),
                height=800,
                scrolling=True,
            )
        else:
            st.warning("Nessun report performance disponibile")


if __name__ == "__main__":
    main()

