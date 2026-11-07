#!/usr/bin/env python3
"""Evaluate an OLS time-of-day baseline with fixed five-fold CV."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold


FEATURE = "time_of_day_hour"
TARGET = "outcome_activity_not_stationary"
SEED = 226
N_SPLITS = 5


def load_data(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or FEATURE not in rows[0] or TARGET not in rows[0]:
        raise ValueError(f"{path} must contain {FEATURE} and {TARGET}")
    x = np.array([[float(row[FEATURE])] for row in rows])
    y = np.array([float(row[TARGET]) for row in rows])
    return rows, x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("Data/exploration_data.csv"),
        help="Development-data CSV used for cross-validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prediction/outputs/time_of_day_baseline"),
    )
    args = parser.parse_args()

    rows, x, y = load_data(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_predictions = np.empty_like(y)
    prediction_rows = []
    fold_metrics = []

    for fold, (train_index, validation_index) in enumerate(splitter.split(x), start=1):
        model = LinearRegression().fit(x[train_index], y[train_index])
        raw_predictions = model.predict(x[validation_index])
        predictions = np.clip(raw_predictions, 0.0, 1.0)
        oof_predictions[validation_index] = predictions
        mae = mean_absolute_error(y[validation_index], predictions)
        rmse = root_mean_squared_error(y[validation_index], predictions)
        fold_metrics.append(
            {
                "fold": fold,
                "training_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "mae": float(mae),
                "rmse": float(rmse),
                "intercept": float(model.intercept_),
                "time_of_day_hour_coefficient": float(model.coef_[0]),
                "out_of_range_predictions_before_clipping": int(
                    np.sum((raw_predictions < 0) | (raw_predictions > 1))
                ),
            }
        )
        for index, raw_prediction, prediction in zip(
            validation_index, raw_predictions, predictions
        ):
            prediction_rows.append(
                {
                    "observation_id": rows[index]["observation_id"],
                    "fold": fold,
                    "actual_not_stationary": f"{y[index]:.6f}",
                    "predicted_not_stationary_raw": f"{raw_prediction:.6f}",
                    "predicted_not_stationary_clipped": f"{prediction:.6f}",
                }
            )

    summary = {
        "model": "ordinary_least_squares_linear_regression",
        "feature": FEATURE,
        "target": TARGET,
        "data": str(args.data),
        "cross_validation": {
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SEED,
        },
        "prediction_postprocessing": "Predictions were clipped to the valid proportion range [0, 1] before scoring.",
        "mean_fold_mae": float(np.mean([fold["mae"] for fold in fold_metrics])),
        "mean_fold_rmse": float(np.mean([fold["rmse"] for fold in fold_metrics])),
        "pooled_out_of_fold_mae": float(mean_absolute_error(y, oof_predictions)),
        "pooled_out_of_fold_rmse": float(root_mean_squared_error(y, oof_predictions)),
        "out_of_range_predictions_before_clipping": int(
            sum(fold["out_of_range_predictions_before_clipping"] for fold in fold_metrics)
        ),
        "fold_metrics": fold_metrics,
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "out_of_fold_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(prediction_rows, key=lambda row: row["observation_id"]))

    summary_path = args.output_dir / "summary.md"
    summary_path.write_text(
        "# Time-of-day OLS baseline\n\n"
        f"- Feature: `{FEATURE}`\n"
        f"- Target: `{TARGET}`\n"
        f"- Cross-validation: {N_SPLITS} folds, shuffled, seed {SEED}\n"
        f"- Mean fold MAE: {summary['mean_fold_mae']:.6f}\n"
        f"- Mean fold RMSE: {summary['mean_fold_rmse']:.6f}\n"
        f"- Pooled out-of-fold MAE: {summary['pooled_out_of_fold_mae']:.6f}\n"
        f"- Pooled out-of-fold RMSE: {summary['pooled_out_of_fold_rmse']:.6f}\n"
        f"- Out-of-range predictions before clipping: {summary['out_of_range_predictions_before_clipping']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
