from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import pygeohash as pgh

COLUMN_ALIASES = {
    "NumberOfLanes": "NumberofLanes",
}

CATEGORICAL_FEATURES = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "day",
    "geohash_hour",
    "RoadType_hour",
    "Weather_hour",
]

NUMERIC_FEATURES = [
    "NumberofLanes",
    "Temperature",
    "hour",
    "minute",
    "timestamp_minutes",
    "is_morning_peak",
    "is_evening_peak",
    "is_peak_hour",
    "latitude",
    "longitude",
]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {source: target for source, target in COLUMN_ALIASES.items() if source in renamed.columns}
    if rename_map:
        renamed = renamed.rename(columns=rename_map)
    return renamed


def _parse_timestamp(value: Any) -> tuple[float, float, float]:
    if pd.isna(value):
        return np.nan, np.nan, np.nan

    text = str(value).strip()
    if not text:
        return np.nan, np.nan, np.nan

    parts = text.split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return np.nan, np.nan, np.nan
        hour = int(parsed.hour)
        minute = int(parsed.minute)

    timestamp_minutes = hour * 60 + minute
    return float(hour), float(minute), float(timestamp_minutes)


@lru_cache(maxsize=8192)
def _decode_geohash(value: str) -> tuple[float, float]:
    decoded = pgh.decode(value)
    if isinstance(decoded, tuple):
        latitude, longitude = decoded
    else:
        latitude = getattr(decoded, "latitude", np.nan)
        longitude = getattr(decoded, "longitude", np.nan)
    return float(latitude), float(longitude)


def _safe_decode_geohash(value: Any) -> tuple[float, float]:
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return np.nan, np.nan

    try:
        return _decode_geohash(text)
    except Exception:
        return np.nan, np.nan


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = standardize_columns(df)

    timestamp_features = frame["timestamp"].apply(_parse_timestamp) if "timestamp" in frame.columns else pd.Series([(np.nan, np.nan, np.nan)] * len(frame), index=frame.index)
    if len(timestamp_features) > 0:
        frame[["hour", "minute", "timestamp_minutes"]] = pd.DataFrame(timestamp_features.tolist(), index=frame.index)
        frame["is_morning_peak"] = frame["hour"].between(7, 10, inclusive="both").astype("float")
        frame["is_evening_peak"] = frame["hour"].between(17, 20, inclusive="both").astype("float")
        frame["is_peak_hour"] = (
            frame["is_morning_peak"].fillna(0).astype(int) | frame["is_evening_peak"].fillna(0).astype(int)
        ).astype("float")

    if "geohash" in frame.columns:
        geo_values = frame["geohash"].apply(_safe_decode_geohash)
        frame[["latitude", "longitude"]] = pd.DataFrame(geo_values.tolist(), index=frame.index)

    if {"geohash", "hour"}.issubset(frame.columns):
        frame["geohash_hour"] = frame["geohash"].astype(str) + "_" + frame["hour"].fillna(-1).astype(int).astype(str)
    if {"RoadType", "hour"}.issubset(frame.columns):
        frame["RoadType_hour"] = frame["RoadType"].astype(str) + "_" + frame["hour"].fillna(-1).astype(int).astype(str)
    if {"Weather", "hour"}.issubset(frame.columns):
        frame["Weather_hour"] = frame["Weather"].astype(str) + "_" + frame["hour"].fillna(-1).astype(int).astype(str)

    return frame


def compute_imputation_stats(df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "categorical_columns": [column for column in CATEGORICAL_FEATURES if column in df.columns],
        "numeric_columns": [column for column in NUMERIC_FEATURES if column in df.columns],
        "numeric_medians": {},
    }

    for column in stats["numeric_columns"]:
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        stats["numeric_medians"][column] = float(numeric_series.median()) if not numeric_series.dropna().empty else 0.0

    return stats


def apply_imputation(df: pd.DataFrame, stats: dict[str, Any]) -> pd.DataFrame:
    frame = df.copy()

    for column in stats.get("categorical_columns", []):
        if column in frame.columns:
            frame[column] = frame[column].fillna("Unknown").astype(str)

    for column, median_value in stats.get("numeric_medians", {}).items():
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(median_value)

    return frame


def prepare_features(df: pd.DataFrame, stats: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    engineered = add_engineered_features(df)
    fitted_stats = stats or compute_imputation_stats(engineered)
    prepared = apply_imputation(engineered, fitted_stats)

    if "timestamp" in prepared.columns:
        prepared = prepared.drop(columns=["timestamp"])

    return prepared, fitted_stats


def get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"demand"}
    return [column for column in df.columns if column not in excluded]
