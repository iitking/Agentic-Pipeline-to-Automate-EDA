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
df_preview = None

if data_source == "Use Sample Dataset":
    sample_file = "data/customer_churn.csv"
    if Path(sample_file).exists():
        dataset_path = sample_file
        st.sidebar.success(f"Loaded: `{sample_file}`")
        df_preview = load_dataset(sample_file)
    else:
        st.sidebar.error("Sample dataset not found.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV, Excel, or Parquet",
        type=["csv", "xlsx", "xls", "parquet"],
    )
    if uploaded_file is not None:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        dataset_path = str(save_path)
        df_preview = load_dataset(dataset_path)
        st.sidebar.success(f"Uploaded: `{uploaded_file.name}`")

if df_preview is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Analysis Objective")
    
    # Auto-detect target column candidate
    schema_quick = detect_schema(df_preview)
    inferred_target = schema_quick.get("inferred_target")
    candidate_name = inferred_target.get("column") if inferred_target else None

    cols = ["None (Unsupervised / General EDA)"] + list(df_preview.columns)
    default_idx = cols.index(candidate_name) if candidate_name in cols else 0

    target_choice = st.sidebar.selectbox(
        "Target Variable (Optional)",
        cols,
        index=default_idx,
        help="Select a target to trigger group comparisons, class balance checks, and hypothesis testing.",
    )
    selected_target = None if target_choice.startswith("None") else target_choice

    user_objective = st.sidebar.text_area(
        "Analysis Goal / Prompt",
        value=f"Perform comprehensive EDA{' focused on predicting ' + selected_target if selected_target else ''}, evaluate data quality, and identify actionable patterns.",
        height=90,
    )

    max_iter = st.sidebar.slider("Max Feedback Iterations", 1, 3, 2)

    # Run button
    run_btn = st.sidebar.button("🚀 Run Agentic EDA", type="primary", use_container_width=True)

    if run_btn and dataset_path:
        with st.status("Running Agentic EDA Workflow...", expanded=True) as status:
            st.write("🔍 **Phase 1:** Profiling dataset and inferring column schemas...")
            st.write("🧠 **Phase 2:** Planning specialized analysis tasks...")
            st.write("🛡️ **Phase 3:** Executing data quality audit (missingness, duplicates, outliers)...")
            st.write("📈 **Phase 4:** Computing statistical distributions and correlation matrices...")
            st.write("🎨 **Phase 5:** Generating interactive Plotly charts...")
            if selected_target:
                st.write(f"🔬 **Phase 6:** Analyzing relationships against target `{selected_target}`...")
            st.write("💡 **Phase 7:** Synthesizing evidence-backed insights and recommendations...")
            st.write("⚖️ **Phase 8:** Validating findings against deterministic calculations...")
            st.write("📑 **Phase 9:** Compiling final report and exporting artifacts...")

            # Run pipeline
            final_state = run_eda_pipeline(
                dataset_path=dataset_path,
                target_column=selected_target,
                user_objective=user_objective,
                max_iterations=max_iter,
            )
            st.session_state["eda_state"] = final_state
            status.update(label="✅ Agentic EDA Analysis Completed Successfully!", state="complete", expanded=False)

    # Render results if present in session state
