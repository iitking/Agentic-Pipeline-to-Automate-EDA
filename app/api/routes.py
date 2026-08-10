"""FastAPI REST API routes for Agentic EDA service."""

import uuid
import shutil
from pathlib import Path
from typing import Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from workflow.graph import run_eda_pipeline

router = APIRouter()

# In-memory job store
JOBS: dict[str, dict[str, Any]] = {}
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AnalyzeRequest(BaseModel):
    file_path: str
    target_column: str | None = None
    user_objective: str = "Perform comprehensive EDA and generate evidence-backed insights."
    max_iterations: int = 2


def run_pipeline_task(job_id: str, file_path: str, target: str | None, objective: str, max_iterations: int):
    """Background execution task."""
    try:
        JOBS[job_id]["status"] = "processing"
        final_state = run_eda_pipeline(
            dataset_path=file_path,
            target_column=target,
            user_objective=objective,
            max_iterations=max_iterations,
        )
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = final_state
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Agentic EDA Pipeline"}


