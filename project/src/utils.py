from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def ensure_directories() -> None:
    for directory in [MODELS_DIR, SUBMISSIONS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def competition_score(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return max(0.0, 100.0 * float(r2_score(y_true, y_pred)))


def save_submission(index_values: pd.Series | np.ndarray, predictions: np.ndarray, output_path: str | Path) -> Path:
    submission = pd.DataFrame({"Index": index_values, "demand": np.clip(np.asarray(predictions), 0.0, None)})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def to_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
