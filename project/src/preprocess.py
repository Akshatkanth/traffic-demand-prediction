from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from feature_engineering import get_model_feature_columns, prepare_features
from utils import PROJECT_ROOT


def resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    fallback = PROJECT_ROOT.parent / "dataset" / candidate.name
    if fallback.exists():
        return fallback

    return candidate


def load_dataset(path: str | Path) -> pd.DataFrame:
    resolved_path = resolve_data_path(path)
    return pd.read_csv(resolved_path)


def build_train_test_frames(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    prepared_train, stats = prepare_features(train_df)
    prepared_test, _ = prepare_features(test_df, stats)

    target = prepared_train["demand"].astype(float)
    feature_columns = [column for column in get_model_feature_columns(prepared_train) if column != "Index"]

    train_features = prepared_train[feature_columns].copy()
    test_features = prepared_test[feature_columns].copy()

    return train_features, target, test_features, stats
