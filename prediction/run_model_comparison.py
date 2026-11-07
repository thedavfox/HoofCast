#!/usr/bin/env python3
"""Tune and compare regression models with fixed five-fold CV.

The script evaluates both feature sets from the Section 6 write-up on the
development data only.  Tuning and scoring use clipped predictions because the
target is a valid proportion in [0, 1].
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


TARGET = "outcome_activity_not_stationary"
SEED = 226
N_SPLITS = 5

WEATHER_NUMERIC = [
    "weather_temperature_c",
    "weather_relative_humidity_pct",
    "weather_precipitation_mm_prior_hour",
    "weather_cloud_cover_pct",
    "weather_shortwave_radiation_w_m2_prior_hour",
    "weather_wind_speed_10m_kmh",
    "weather_wind_gusts_10m_kmh",
    "weather_is_day",
]
VIDEO_NUMERIC = [
    "pre_posture_standing",
    "pre_posture_lying_sternal",
    "pre_posture_lying_lateral",
    "pre_posture_transitioning",
    "pre_posture_unknown",
    "pre_visibility_full",
    "pre_visibility_partial",
    "pre_visibility_not_visible",
    "pre_transition_stand_to_lie",
    "pre_transition_lie_to_stand",
    "pre_transition_none",
    "pre_transition_unknown",
    "pre_pawing",
    "pre_biting_object",
    "pre_rubbing",
    "pre_repetitive_movement",
    "pre_no_secondary_behavior",
]
POSTURE_CATEGORICAL = "pre_posture_dominant"


def clipped_predictions(estimator, x):
    return np.clip(estimator.predict(x), 0.0, 1.0)


def neg_clipped_mae(estimator, x, y):
    return -mean_absolute_error(y, clipped_predictions(estimator, x))


def neg_clipped_rmse(estimator, x, y):
    return -root_mean_squared_error(y, clipped_predictions(estimator, x))


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    needed = set(WEATHER_NUMERIC + VIDEO_NUMERIC + [POSTURE_CATEGORICAL, TARGET])
    missing = needed.difference(rows[0]) if rows else needed
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return rows


def make_feature_matrix(rows, feature_set):
    hours = np.array([float(row["time_of_day_hour"]) for row in rows])
    wind_degrees = np.array([float(row["weather_wind_direction_10m_degrees"]) for row in rows])
    time_features = np.column_stack(
        [np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)]
    )
    wind_features = np.column_stack(
        [
            np.sin(2 * np.pi * wind_degrees / 360),
            np.cos(2 * np.pi * wind_degrees / 360),
        ]
    )
    weather = np.array([[float(row[column]) for column in WEATHER_NUMERIC] for row in rows])
    numeric = np.column_stack([time_features, weather, wind_features])
    numeric_names = ["time_of_day_sin", "time_of_day_cos"] + WEATHER_NUMERIC + [
        "wind_direction_sin",
        "wind_direction_cos",
    ]
    if feature_set == "full":
        video = np.array([[float(row[column]) for column in VIDEO_NUMERIC] for row in rows])
        numeric = np.column_stack([numeric, video])
        numeric_names += VIDEO_NUMERIC
        posture = np.array([row[POSTURE_CATEGORICAL] for row in rows], dtype=object).reshape(-1, 1)
        return np.column_stack([numeric, posture]), numeric_names, True
    if feature_set == "environmental_temporal":
        return numeric, numeric_names, False
    raise ValueError(f"Unknown feature set: {feature_set}")


def make_pipeline(model, numeric_count, has_categorical, standardize):
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if standardize:
        numeric_steps.append(("scale", StandardScaler()))
    transformers = [("numeric", Pipeline(numeric_steps), list(range(numeric_count)))]
    if has_categorical:
        transformers.append(
            (
                "posture",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                [numeric_count],
            )
        )
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", model)])


def model_definitions():
    return {
        "OLS": {
            "model": LinearRegression(),
            "grid": [{}],
            "standardize": True,
        },
        "Elastic Net": {
            "model": ElasticNet(max_iter=10000, random_state=SEED),
            "grid": {
                "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95, 1.0],
            },
            "standardize": True,
        },
        "RBF SVR": {
            "model": SVR(kernel="rbf"),
            "grid": {
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.01, 0.1, 1.0],
                "model__epsilon": [0.01, 0.05, 0.1],
            },
            "standardize": True,
        },
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
            "grid": {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__max_features": [0.5, "sqrt", 1.0],
            },
            "standardize": False,
        },
    }


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_configuration(name, definition, feature_set, x, y, rows, splitter):
    pipeline = make_pipeline(
        definition["model"], x.shape[1] - (1 if feature_set == "full" else 0),
        feature_set == "full", definition["standardize"],
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=definition["grid"],
        scoring={"mae": neg_clipped_mae, "rmse": neg_clipped_rmse},
        refit="mae",
        cv=splitter,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(x, y)
    selected = {key.replace("model__", ""): json_value(value) for key, value in search.best_params_.items()}
    prediction_rows, fold_metrics = [], []
    all_actual, all_predictions = [], []
    for fold, (train_index, validation_index) in enumerate(splitter.split(x, y), start=1):
        estimator = clone(search.best_estimator_).fit(x[train_index], y[train_index])
        raw_predictions = estimator.predict(x[validation_index])
        predictions = np.clip(raw_predictions, 0.0, 1.0)
        actual = y[validation_index]
        all_actual.extend(actual)
        all_predictions.extend(predictions)
        fold_metrics.append(
            {
                "fold": fold,
                "mae": float(mean_absolute_error(actual, predictions)),
                "rmse": float(root_mean_squared_error(actual, predictions)),
                "out_of_range_before_clipping": int(np.sum((raw_predictions < 0) | (raw_predictions > 1))),
            }
        )
        for index, raw, prediction in zip(validation_index, raw_predictions, predictions):
            prediction_rows.append(
                {
                    "observation_id": rows[index]["observation_id"],
                    "algorithm": name,
                    "feature_set": feature_set,
                    "fold": fold,
                    "actual_not_stationary": f"{y[index]:.6f}",
                    "prediction_raw": f"{raw:.6f}",
                    "prediction_clipped": f"{prediction:.6f}",
                }
            )
    return {
        "algorithm": name,
        "feature_set": feature_set,
        "selected_tuning": selected,
        "tuning_candidates": int(len(search.cv_results_["params"])),
        "mean_fold_mae": float(np.mean([item["mae"] for item in fold_metrics])),
        "mean_fold_rmse": float(np.mean([item["rmse"] for item in fold_metrics])),
        "pooled_out_of_fold_mae": float(mean_absolute_error(all_actual, all_predictions)),
        "pooled_out_of_fold_rmse": float(root_mean_squared_error(all_actual, all_predictions)),
        "out_of_range_before_clipping": int(sum(item["out_of_range_before_clipping"] for item in fold_metrics)),
        "fold_metrics": fold_metrics,
        "predictions": prediction_rows,
    }


def latex_tuning(result):
    if not result["selected_tuning"]:
        return "None"
    parts = []
    for key, value in result["selected_tuning"].items():
        rendered_key = key.replace("_", "\\_")
        rendered = str(value).replace("_", "\\_")
        parts.append(f"{rendered_key}={rendered}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Data/exploration_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("prediction/outputs/model_comparison"))
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("prediction/outputs/time_of_day_baseline/metrics.json"),
    )
    args = parser.parse_args()
    rows = load_rows(args.data)
    y = np.array([float(row[TARGET]) for row in rows])
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    results = []
    all_predictions = []
    for feature_set in ["environmental_temporal", "full"]:
        x, feature_names, has_categorical = make_feature_matrix(rows, feature_set)
        for name, definition in model_definitions().items():
            result = evaluate_configuration(name, definition, feature_set, x, y, rows, splitter)
            result["feature_count_before_one_hot_encoding"] = len(feature_names) + int(has_categorical)
            all_predictions.extend(result.pop("predictions"))
            results.append(result)
            print(f"{name} / {feature_set}: MAE={result['mean_fold_mae']:.6f}, RMSE={result['mean_fold_rmse']:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.baseline_metrics.open(encoding="utf-8") as file:
        baseline = json.load(file)
    summary = {
        "data": str(args.data),
        "target": TARGET,
        "cross_validation": {"n_splits": N_SPLITS, "shuffle": True, "random_state": SEED},
        "prediction_postprocessing": "All predictions were clipped to [0, 1] before scoring.",
        "baseline": baseline,
        "comparisons": results,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "out_of_fold_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_predictions[0].keys())
        writer.writeheader()
        writer.writerows(sorted(all_predictions, key=lambda row: (row["algorithm"], row["feature_set"], row["observation_id"])))
    with (args.output_dir / "validation_results.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["algorithm", "feature_set", "selected_tuning", "tuning_candidates", "mean_fold_mae", "mean_fold_rmse", "pooled_out_of_fold_mae", "pooled_out_of_fold_rmse", "out_of_range_before_clipping"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({**{key: result[key] for key in fields if key in result}, "selected_tuning": json.dumps(result["selected_tuning"], sort_keys=True)})
    latex_newline = r"\\"
    latex_rows = [
        "\\begin{table}[H]", "\\centering\\small", "\\begin{tabular}{llcc}", "\\toprule",
        f"Algorithm & Feature set & Selected tuning & Validation metric {latex_newline}", "\\midrule",
        f"Baseline & --- & --- & MAE = {baseline['mean_fold_mae']:.3f}; RMSE = {baseline['mean_fold_rmse']:.3f} {latex_newline}",
    ]
    labels = {"environmental_temporal": "Set A", "full": "Set B"}
    for result in results:
        latex_rows.append(
            f"{result['algorithm']} & {labels[result['feature_set']]} & {latex_tuning(result)} & MAE = {result['mean_fold_mae']:.3f}; RMSE = {result['mean_fold_rmse']:.3f} {latex_newline}")
    latex_rows += ["\\bottomrule", "\\end{tabular}", "\\caption{Validation performance by algorithm and feature set.}", "\\end{table}", ""]
    (args.output_dir / "validation_table.tex").write_text("\n".join(latex_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
