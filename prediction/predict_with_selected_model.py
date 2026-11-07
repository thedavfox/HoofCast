#!/usr/bin/env python3
"""Generate clipped non-stationary-activity predictions for an input CSV."""

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from run_model_comparison import (
    POSTURE_CATEGORICAL,
    VIDEO_NUMERIC,
    WEATHER_NUMERIC,
    make_feature_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the selected Set B RBF-SVR model to a CSV file."
    )
    parser.add_argument("input_csv", type=Path, help="CSV with the required Set B raw columns.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("prediction/selected_rbf_svr_set_b.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prediction/outputs/predictions.csv"),
    )
    args = parser.parse_args()

    # load_rows validates target as well, but the prediction input does not need
    # the outcome.  Add a temporary placeholder target solely for validation.
    with args.input_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("The input CSV has no data rows.")
    required = {
        "time_of_day_hour",
        "weather_wind_direction_10m_degrees",
        *WEATHER_NUMERIC,
        *VIDEO_NUMERIC,
        POSTURE_CATEGORICAL,
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    for row in raw_rows:
        row.setdefault("outcome_activity_not_stationary", "0")
    x, _, _ = make_feature_matrix(raw_rows, "full")
    model = joblib.load(args.model)
    raw_predictions = model.predict(x)
    predictions = np.clip(raw_predictions, 0.0, 1.0)

    id_column = "observation_id" if "observation_id" in fieldnames else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = ([id_column] if id_column else []) + [
        "prediction_raw_not_stationary",
        "prediction_not_stationary",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        for index, (row, raw, prediction) in enumerate(zip(raw_rows, raw_predictions, predictions), start=1):
            result = {
                "prediction_raw_not_stationary": f"{raw:.6f}",
                "prediction_not_stationary": f"{prediction:.6f}",
            }
            if id_column:
                result[id_column] = row[id_column]
            writer.writerow(result)
    print(f"Saved {len(raw_rows)} predictions to {args.output}.")


if __name__ == "__main__":
    main()
