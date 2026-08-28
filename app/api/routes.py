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


@router.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset (CSV, XLSX, Parquet)."""
    allowed_exts = [".csv", ".xlsx", ".xls", ".parquet", ".pq"]
    ext = Path(file.filename or "").suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(allowed_exts)}",
        )

    file_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "File uploaded successfully",
        "file_path": str(save_path),
        "filename": file.filename,
        "size_bytes": save_path.stat().st_size,
    }


@router.post("/api/analyze")
def start_analysis(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Start an EDA analysis job."""
    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found at: {req.file_path}")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "file_path": req.file_path,
        "target_column": req.target_column,
        "result": None,
        "error": None,
    }

    background_tasks.add_task(
        run_pipeline_task,
        job_id,
        req.file_path,
        req.target_column,
        req.user_objective,
        req.max_iterations,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Analysis started in background.",
    }


@router.get("/api/status/{job_id}")
def get_job_status(job_id: str):
    """Check status of an analysis job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
    }


@router.get("/api/report/{job_id}")
def get_job_report(job_id: str, format: str = "json"):
    """Get report result as JSON or HTML."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]
    if job["status"] != "completed":
        return {"status": job["status"], "message": "Analysis not yet completed"}

    result = job["result"]
    if format.lower() == "html":
        report_path = result.get("final_report_path")
        if report_path and Path(report_path).exists():
            with open(report_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        raise HTTPException(status_code=404, detail="HTML report file missing")

    return {
        "job_id": job_id,
        "dataset_name": result.get("dataset_name"),
        "quality_score": result.get("quality_score"),
        "insights": result.get("validated_insights"),
        "recommendations": result.get("actionable_recommendations"),
        "warnings": result.get("warnings"),
        "report_path": result.get("final_report_path"),
    }


@router.get("/api/artifacts/{filename}")
def download_artifact(filename: str):
    """Download a generated artifact from artifacts directory."""
    path = Path("artifacts") / filename
    if not path.exists():
        path = Path("artifacts/reports") / filename
    if not path.exists():
        path = Path("artifacts/charts") / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    return FileResponse(path)
