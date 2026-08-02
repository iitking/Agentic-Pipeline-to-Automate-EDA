"""Streamlit Interactive Web Dashboard for Agentic EDA."""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when launched directly via streamlit run
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:

import streamlit as st
import pandas as pd
import plotly.io as pio

from workflow.graph import run_eda_pipeline
from tools.profiling import load_dataset, detect_schema

# Set page config
st.set_page_config(
    page_title="Agentic EDA Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Agentic Pipeline to Automate EDA")
st.markdown(
    "Adaptive Exploratory Data Analysis combining deterministic Python analytics "
    "with LLM planning, evidence-backed interpretation, and publication-ready reporting."
)

# Sidebar configurations
st.sidebar.header("⚙️ Configuration")

data_source = st.sidebar.radio(
    "Dataset Source",
    ["Use Sample Dataset", "Upload My Dataset"],
)

dataset_path = None
