"""Report generation module compiling state artifacts into HTML and JSON exports."""

from pathlib import Path
from typing import Any
import json
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from workflow.state import EDAState


class ReportGenerator:
    """Renders comprehensive EDA reports and saves machine-readable artifacts."""

    def __init__(self, templates_dir: str | Path | None = None, output_dir: str | Path = "artifacts"):
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        self.env = Environment(loader=FileSystemLoader(str(self.templates_dir)), autoescape=True)

    def generate_report(self, state: EDAState) -> str:
        """Render HTML report and write all summary artifacts."""
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save dataset_profile.json
        profile_path = self.output_dir / "dataset_profile.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(state.get("data_profile", {}), f, indent=2, default=str)
