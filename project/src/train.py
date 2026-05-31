from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder

from src.feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.preprocess import build_train_test_frames, load_dataset
from src.utils import DATA_DIR, MODELS_DIR, SUBMISSIONS_DIR, competition_score, ensure_directories, save_bundle, save_submission, set_seed, to_json

try:
    from catboost import CatBoostRegressor, Pool
except Exception:  # pragma: no cover - optional dependency
    CatBoostRegressor = None
    Pool = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


def build_ordinal_preprocessor(categorical_columns: list[str], numeric_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            ("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def cross_validate_pipeline_model(
    model_factory: Callable[[], Any],
    train_features: pd.DataFrame,
    target: pd.Series,
    test_features: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[float], list[Any]]:
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_predictions = np.zeros(len(train_features))
    test_predictions = np.zeros(len(test_features))
    fold_scores: list[float] = []
    fitted_pipelines: list[Any] = []

    for fold_number, (train_index, valid_index) in enumerate(folds.split(train_features, target), start=1):
        x_train = train_features.iloc[train_index]
        y_train = target.iloc[train_index]
        x_valid = train_features.iloc[valid_index]
        y_valid = target.iloc[valid_index]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_ordinal_preprocessor(categorical_columns, numeric_columns)),
                ("model", model_factory()),
            ]
        )
        pipeline.fit(x_train, y_train)
        valid_predictions = pipeline.predict(x_valid)
        oof_predictions[valid_index] = valid_predictions
        test_predictions += pipeline.predict(test_features) / n_splits
        fold_score = float(max(0.0, 100.0 * r2_score(y_valid, valid_predictions)))
        fold_scores.append(fold_score)
        fitted_pipelines.append(pipeline)
        print(f"Fold {fold_number}: R2={fold_score / 100.0:.6f}")

    return oof_predictions, test_predictions, fold_scores, fitted_pipelines


def train_random_forest(train_features: pd.DataFrame, target: pd.Series, test_features: pd.DataFrame) -> dict[str, Any]:
    categorical_columns = [column for column in CATEGORICAL_FEATURES if column in train_features.columns]
    numeric_columns = [column for column in NUMERIC_FEATURES if column in train_features.columns]

    def factory() -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )

    oof_predictions, test_predictions, fold_scores, fitted_models = cross_validate_pipeline_model(
        factory,
        train_features,
        target,
        test_features,
        categorical_columns,
        numeric_columns,
    )
    return {
        "model_name": "random_forest",
        "oof_predictions": oof_predictions,
        "test_predictions": test_predictions,
        "fold_scores": fold_scores,
        "mean_score": float(np.mean(fold_scores)),
        "std_score": float(np.std(fold_scores)),
        "model": fitted_models[-1],
    }


def train_lightgbm(train_features: pd.DataFrame, target: pd.Series, test_features: pd.DataFrame) -> dict[str, Any]:
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed")

    categorical_columns = [column for column in CATEGORICAL_FEATURES if column in train_features.columns]
    numeric_columns = [column for column in NUMERIC_FEATURES if column in train_features.columns]

    def factory() -> LGBMRegressor:
        return LGBMRegressor(
            n_estimators=4000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

    return _train_boosting_model(factory, train_features, target, test_features, categorical_columns, numeric_columns, model_name="lightgbm")


def train_xgboost(train_features: pd.DataFrame, target: pd.Series, test_features: pd.DataFrame) -> dict[str, Any]:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed")

    categorical_columns = [column for column in CATEGORICAL_FEATURES if column in train_features.columns]
    numeric_columns = [column for column in NUMERIC_FEATURES if column in train_features.columns]

    def factory() -> XGBRegressor:
        return XGBRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    return _train_boosting_model(factory, train_features, target, test_features, categorical_columns, numeric_columns, model_name="xgboost")


def _train_boosting_model(
    model_factory: Callable[[], Any],
    train_features: pd.DataFrame,
    target: pd.Series,
    test_features: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    model_name: str,
) -> dict[str, Any]:
    oof_predictions, test_predictions, fold_scores, fitted_models = cross_validate_pipeline_model(
        model_factory,
        train_features,
        target,
        test_features,
        categorical_columns,
        numeric_columns,
    )
    return {
        "model_name": model_name,
        "oof_predictions": oof_predictions,
        "test_predictions": test_predictions,
        "fold_scores": fold_scores,
        "mean_score": float(np.mean(fold_scores)),
        "std_score": float(np.std(fold_scores)),
        "model": fitted_models[-1],
    }


def fit_final_pipeline_model(
    model_factory: Callable[[], Any],
    train_features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> Any:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_ordinal_preprocessor(categorical_columns, numeric_columns)),
            ("model", model_factory()),
        ]
    )
    pipeline.fit(train_features, target)
    return pipeline


def fit_final_catboost(train_features: pd.DataFrame, target: pd.Series) -> Any:
    if CatBoostRegressor is None or Pool is None:
        raise ImportError("catboost is not installed")

    categorical_columns = [column for column in CATEGORICAL_FEATURES if column in train_features.columns]
    cat_feature_indices = [train_features.columns.get_loc(column) for column in categorical_columns]

    full_train = train_features.copy()
    for column in categorical_columns:
        full_train[column] = full_train[column].astype(str)

    train_pool = Pool(full_train, target, cat_features=cat_feature_indices)
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="R2",
        random_seed=42,
        verbose=200,
        allow_writing_files=False,
    )
    model.fit(train_pool)
    return model


def train_catboost(train_features: pd.DataFrame, target: pd.Series, test_features: pd.DataFrame) -> dict[str, Any]:
    if CatBoostRegressor is None or Pool is None:
        raise ImportError("catboost is not installed")

    categorical_columns = [column for column in CATEGORICAL_FEATURES if column in train_features.columns]
    cat_feature_indices = [train_features.columns.get_loc(column) for column in categorical_columns]

    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    n_splits = folds.get_n_splits()
    oof_predictions = np.zeros(len(train_features))
    test_predictions = np.zeros(len(test_features))
    fold_scores: list[float] = []
    fitted_models: list[Any] = []

    for fold_number, (train_index, valid_index) in enumerate(folds.split(train_features, target), start=1):
        x_train = train_features.iloc[train_index].copy()
        y_train = target.iloc[train_index]
        x_valid = train_features.iloc[valid_index].copy()
        y_valid = target.iloc[valid_index]
        test_frame = test_features.copy()

        for column in categorical_columns:
            x_train[column] = x_train[column].astype(str)
            x_valid[column] = x_valid[column].astype(str)
            test_frame[column] = test_frame[column].astype(str)

        train_pool = Pool(x_train, y_train, cat_features=cat_feature_indices)
        valid_pool = Pool(x_valid, y_valid, cat_features=cat_feature_indices)

        model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            eval_metric="R2",
            random_seed=42,
            verbose=200,
            od_type="Iter",
            od_wait=200,
            allow_writing_files=False,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        valid_predictions = model.predict(x_valid)
        oof_predictions[valid_index] = valid_predictions
        test_predictions += model.predict(test_frame) / n_splits
        fold_score = float(max(0.0, 100.0 * r2_score(y_valid, valid_predictions)))
        fold_scores.append(fold_score)
        fitted_models.append(model)
        print(f"Fold {fold_number}: R2={fold_score / 100.0:.6f}")

    return {
        "model_name": "catboost",
        "oof_predictions": oof_predictions,
        "test_predictions": test_predictions,
        "fold_scores": fold_scores,
        "mean_score": float(np.mean(fold_scores)),
        "std_score": float(np.std(fold_scores)),
        "model": fitted_models[-1],
        "categorical_columns": categorical_columns,
    }

def train_model(model_name: str, train_path: Path, test_path: Path) -> dict[str, Any]:
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)
    train_features, target, test_features, stats = build_train_test_frames(train_df, test_df)

    if model_name == "catboost":
        result = train_catboost(train_features, target, test_features)
        result["model"] = fit_final_catboost(train_features, target)
    elif model_name == "lightgbm":
        result = train_lightgbm(train_features, target, test_features)
        categorical_columns = [column for column in CATEGORICAL_FEATURES if column in train_features.columns]
        numeric_columns = [column for column in NUMERIC_FEATURES if column in train_features.columns]
        result["model"] = fit_final_pipeline_model(
            lambda: LGBMRegressor(
                n_estimators=4000,
                learning_rate=0.03,
                num_leaves=63,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            ),
            train_features,
            target,
            categorical_columns,
            numeric_columns,
        )
    elif model_name == "xgboost":
        result = train_xgboost(train_features, target, test_features)
        categorical_columns = [column for column in CATEGORICAL_FEATURES if column in train_features.columns]
        numeric_columns = [column for column in NUMERIC_FEATURES if column in train_features.columns]
        result["model"] = fit_final_pipeline_model(
            lambda: XGBRegressor(
                n_estimators=3000,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            ),
            train_features,
            target,
            categorical_columns,
            numeric_columns,
        )
    else:
        result = train_random_forest(train_features, target, test_features)
        categorical_columns = [column for column in CATEGORICAL_FEATURES if column in train_features.columns]
        numeric_columns = [column for column in NUMERIC_FEATURES if column in train_features.columns]
        result["model"] = fit_final_pipeline_model(
            lambda: RandomForestRegressor(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1,
            ),
            train_features,
            target,
            categorical_columns,
            numeric_columns,
        )

    result["feature_stats"] = stats
    result["train_features"] = train_features
    result["test_features"] = test_features
    result["target"] = target
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train traffic demand models")
    parser.add_argument("--model", choices=["catboost", "lightgbm", "xgboost", "random_forest"], default="catboost")
    parser.add_argument("--train-path", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--test-path", type=Path, default=DATA_DIR / "test.csv")
    parser.add_argument("--submission-path", type=Path, default=SUBMISSIONS_DIR / "submission.csv")
    parser.add_argument("--artifact-path", type=Path, default=MODELS_DIR / "model_bundle.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    ensure_directories()

    result = train_model(args.model, args.train_path, args.test_path)
    train_features = result.pop("train_features")
    test_features = result.pop("test_features")
    target = result.pop("target")
    fold_scores = result["fold_scores"]

    train_frame = load_dataset(args.train_path)
    test_frame = load_dataset(args.test_path)
    _, _, _, _ = build_train_test_frames(train_frame, test_frame)

    train_score = competition_score(target, result["oof_predictions"])
    print(f"Mean fold score: {np.mean(fold_scores):.6f}")
    print(f"Std fold score: {np.std(fold_scores):.6f}")
    print(f"OOF competition score: {train_score:.6f}")

    save_submission(test_frame["Index"], result["test_predictions"], args.submission_path)
    save_bundle(
        {
            "model_name": args.model,
            "model": result["model"],
            "feature_columns": list(train_features.columns),
            "feature_stats": result["feature_stats"],
            "categorical_columns": [column for column in CATEGORICAL_FEATURES if column in train_features.columns],
        },
        args.artifact_path,
    )
    to_json(
        MODELS_DIR / f"{args.model}_metrics.json",
        {
            "model": args.model,
            "fold_scores": fold_scores,
            "mean_fold_score": float(np.mean(fold_scores)),
            "std_fold_score": float(np.std(fold_scores)),
            "oof_competition_score": train_score,
        },
    )


if __name__ == "__main__":
    main()
