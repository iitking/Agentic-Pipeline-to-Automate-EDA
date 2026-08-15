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
    print(f"=======================================================\n")

    result = run_eda_pipeline(
        dataset_path=dataset_path,
        target_column=args.target,
        user_objective=args.objective,
        max_iterations=args.iterations,
    )

    print("\n-------------------------------------------------------")
    print(f"✅ EDA Completed Successfully!")
    print(f"📊 Dataset Shape: {result.get('data_profile', {}).get('rows')} rows, {result.get('data_profile', {}).get('columns')} columns")
    print(f"🛡️ Data Quality Score: {result.get('quality_score')}/100")
    print(f"💡 Key Insights Generated: {len(result.get('validated_insights', []))}")
    print(f"📑 Report Path: {result.get('final_report_path')}")
    print("-------------------------------------------------------\n")

    for i, ins in enumerate(result.get("validated_insights", []), 1):
        print(f"{i}. [{ins.get('category')}] {ins.get('headline')}")
        print(f"   {ins.get('observation')}")

    print(f"\nView full interactive HTML report at:\nfile://{Path(result.get('final_report_path', '')).resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Agentic Pipeline to Automate EDA")
    parser.add_argument("--file", "-f", type=str, help="Path to CSV, Excel, or Parquet file")
    parser.add_argument("--target", "-t", type=str, default=None, help="Target variable name (optional)")
    parser.add_argument("--objective", "-o", type=str, default="Perform comprehensive EDA, identify data quality problems, and generate verified insights.", help="Analysis objective")
    parser.add_argument("--iterations", type=int, default=2, help="Max planning feedback iterations")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI REST backend server")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit interactive web dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for FastAPI server")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")

