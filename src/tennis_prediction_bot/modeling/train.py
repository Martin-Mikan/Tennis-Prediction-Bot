from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from tennis_prediction_bot.config.settings import configure_logging, get_settings
from tennis_prediction_bot.data.cleaning import clean_raw_data
from tennis_prediction_bot.features.builder import build_features
from tennis_prediction_bot.utils.io import ensure_directory, write_json


LOGGER = logging.getLogger(__name__)
NUMERIC_COLUMNS = [
    "player_rank",
    "opponent_rank",
    "rank_diff",
    "player_rank_points",
    "opponent_rank_points",
    "rank_points_diff",
    "player_age",
    "opponent_age",
    "age_diff",
    "player_height",
    "opponent_height",
    "height_diff",
    "player_elo",
    "opponent_elo",
    "elo_diff",
    "player_surface_elo",
    "opponent_surface_elo",
    "surface_elo_diff",
    "player_matches_before",
    "opponent_matches_before",
    "player_win_rate",
    "opponent_win_rate",
    "recent_form_diff",
    "player_recent_win_rate",
    "opponent_recent_win_rate",
    "player_surface_win_rate",
    "opponent_surface_win_rate",
    "surface_win_rate_diff",
    "h2h_wins",
    "h2h_losses",
    "h2h_diff",
]
CATEGORICAL_COLUMNS = [
    "surface",
    "tourney_level",
    "player_hand",
    "opponent_hand",
    "handedness_matchup",
]
TARGET_COLUMN = "label"


def _split_data(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = features.loc[features["season"] < 2022].copy()
    validation = features.loc[features["season"].between(2022, 2023)].copy()
    test = features.loc[features["season"] >= 2024].copy()
    return train, validation, test


def _build_baseline_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1500)),
        ]
    )


def _build_boosted_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    boosted = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=50,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", CalibratedClassifierCV(boosted, method="sigmoid", cv=3)),
        ]
    )


def _build_lightgbm_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    lightgbm = LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        verbose=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", lightgbm),
        ]
    )


def _evaluate(model: Pipeline, frame: pd.DataFrame) -> dict[str, float]:
    X = frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = frame[TARGET_COLUMN]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
    }


def train_model(features: pd.DataFrame, artifacts_dir: Path) -> dict[str, object]:
    ensure_directory(artifacts_dir)
    train_frame, validation_frame, test_frame = _split_data(features)

    candidate_models = {
        "logistic_regression": _build_baseline_pipeline(),
        "calibrated_hist_gradient_boosting": _build_boosted_pipeline(),
        "lightgbm": _build_lightgbm_pipeline(),
    }
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    fitted_models: dict[str, Pipeline] = {}

    X_train = train_frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y_train = train_frame[TARGET_COLUMN]

    for name, pipeline in candidate_models.items():
        LOGGER.info("Training model: %s", name)
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        metrics[name] = {
            "validation": _evaluate(pipeline, validation_frame),
            "test": _evaluate(pipeline, test_frame),
        }

    best_model_name = max(
        metrics,
        key=lambda model_name: (
            metrics[model_name]["validation"]["roc_auc"],
            -metrics[model_name]["validation"]["log_loss"],
        ),
    )
    best_model = fitted_models[best_model_name]

    model_path = artifacts_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)

    metadata = {
        "best_model_name": best_model_name,
        "metrics": metrics,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "test_rows": int(len(test_frame)),
        "train_seasons": sorted(train_frame["season"].unique().tolist()),
        "validation_seasons": sorted(validation_frame["season"].unique().tolist()),
        "test_seasons": sorted(test_frame["season"].unique().tolist()),
    }
    write_json(artifacts_dir / "model_metadata.json", metadata)
    write_json(artifacts_dir / "evaluation_metrics.json", metrics)
    return {
        "model_path": model_path,
        "metadata": metadata,
    }


def run_training_pipeline() -> dict[str, object]:
    settings = get_settings()
    configure_logging(settings.log_level)

    LOGGER.info("Cleaning raw data from %s", settings.raw_data_dir)
    cleaned_data = clean_raw_data(settings.raw_data_dir, settings.processed_data_dir)

    LOGGER.info("Building training features")
    feature_outputs = build_features(cleaned_data, settings.processed_data_dir)

    LOGGER.info("Training models")
    training_outputs = train_model(feature_outputs["features"], settings.artifacts_dir)

    summary = {
        "cleaned_match_rows": int(len(cleaned_data["matches"])),
        "feature_rows": int(len(feature_outputs["features"])),
        "player_snapshots": int(len(feature_outputs["player_snapshots"])),
        "best_model_name": training_outputs["metadata"]["best_model_name"],
    }
    write_json(settings.artifacts_dir / "training_summary.json", summary)
    return summary


def main() -> None:
    summary = run_training_pipeline()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
