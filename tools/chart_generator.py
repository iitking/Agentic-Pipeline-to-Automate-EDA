
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def ensure_output_dir(dir_path: str | Path) -> Path:
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


