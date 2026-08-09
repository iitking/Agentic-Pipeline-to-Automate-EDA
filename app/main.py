"""Application entry point supporting CLI, FastAPI server, and Streamlit UI."""

import argparse
import sys
import subprocess
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from workflow.graph import run_eda_pipeline

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Pipeline to Automate EDA",
    description="Agentic EDA system combining reliable Python analytics with LLM planning and evidence validation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


def run_cli(args):
    """Run EDA directly from command line."""
    dataset_path = args.file
    if not dataset_path:
        print("Error: Please provide a dataset file path via --file <path>")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"🚀 Running Agentic EDA on: {dataset_path}")
    if args.target:
        print(f"🎯 Target Variable: {args.target}")
