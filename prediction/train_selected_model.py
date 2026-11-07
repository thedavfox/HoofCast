#!/usr/bin/env python3
"""Train and save the final selected horse-activity prediction model.

The model was selected with the procedure in ``run_model_comparison.py``:
five-fold cross-validation on the 80% development dataset, with MAE as the
primary selection criterion.  The chosen configuration is the Set B RBF SVR.
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVR

from run_model_comparison import (
    N_SPLITS,
    POSTURE_CATEGORICAL,
    SEED,
    TARGET,
    VIDEO_NUMERIC,
    WEATHER_NUMERIC,
    load_rows,
    make_feature_matrix,
    make_pipeline,
)


# Selected from the 48-candidate Set B RBF-SVR grid by the lowest mean
# five-fold validation MAE.  C controls the penalty for errors; gamma controls
# the locality of the RBF kernel; epsilon is the no-penalty error margin.
SELECTED_TUNING = {"C": 1.0, "gamma": 0.01, "epsilon": 0.01}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Data/exploration_data.csv"))
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("prediction/selected_rbf_svr_set_b.pkl"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("prediction/selected_rbf_svr_set_b_metadata.json"),
    )
    args = parser.parse_args()

    rows = load_rows(args.data)
    y = np.array([float(row[TARGET]) for row in rows])

    # Set B consists of the following feature groups:
    #   * circular hour-of-day features (sin/cos),
    #   * weather features, including circular wind direction (sin/cos), and
    #   * pre-period video behavior features plus one-hot dominant posture.
    x, numeric_names, has_categorical = make_feature_matrix(rows, "full")
    model = SVR(kernel="rbf", **SELECTED_TUNING)
    pipeline = make_pipeline(
        model=model,
        numeric_count=len(numeric_names),
        has_categorical=has_categorical,
        standardize=True,
    )
    pipeline.fit(x, y)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_output)

    metadata = {
        "model_name": "RBF-kernel support vector regression (Set B)",
        "target": TARGET,
        "training_data": str(args.data),
        "training_observations": len(rows),
        "selection_procedure": {
            "cv": f"{N_SPLITS}-fold shuffled KFold",
            "random_seed": SEED,
            "primary_metric": "mean validation MAE",
        },
        "selected_tuning": SELECTED_TUNING,
        "raw_required_columns": [
            "time_of_day_hour",
            "weather_wind_direction_10m_degrees",
            *WEATHER_NUMERIC,
            *VIDEO_NUMERIC,
            POSTURE_CATEGORICAL,
        ],
        "derived_features": [
            "sin(2*pi*time_of_day_hour/24)",
            "cos(2*pi*time_of_day_hour/24)",
            "sin(2*pi*weather_wind_direction_10m_degrees/360)",
            "cos(2*pi*weather_wind_direction_10m_degrees/360)",
        ],
        "prediction_range": "Raw predictions should be clipped to [0, 1].",
    }
    with args.metadata_output.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    print(f"Saved {args.model_output} ({len(rows)} development observations).")
    print(f"Saved {args.metadata_output}.")


if __name__ == "__main__":
    main()
