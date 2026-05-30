from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_engineering import prepare_features
from preprocess import load_dataset
from utils import DATA_DIR, SUBMISSIONS_DIR, load_bundle, save_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate predictions from a saved model bundle")
    parser.add_argument("--test-path", type=Path, default=DATA_DIR / "test.csv")
    parser.add_argument("--artifact-path", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "model_bundle.joblib")
    parser.add_argument("--output-path", type=Path, default=SUBMISSIONS_DIR / "submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_frame = load_dataset(args.test_path)
    bundle = load_bundle(args.artifact_path)
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", [])
    feature_stats = bundle.get("feature_stats")

    prepared_test, _ = prepare_features(test_frame, feature_stats)
    if feature_columns:
        prepared_test = prepared_test[feature_columns]

    if hasattr(model, "predict"):
        predictions = model.predict(prepared_test)
    else:
        raise TypeError("Loaded artifact does not expose predict()")

    save_submission(test_frame["Index"], predictions, args.output_path)
    print(f"Saved submission to {args.output_path}")


if __name__ == "__main__":
    main()
