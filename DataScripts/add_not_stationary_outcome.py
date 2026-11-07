#!/usr/bin/env python3
"""Add outcome_activity_not_stationary to the project observation CSVs.

The derived outcome is the complement of the final-window stationary proportion:
outcome_activity_not_stationary = 1 - outcome_activity_stationary.
"""

import argparse
import csv
from pathlib import Path


TARGET = "outcome_activity_not_stationary"
SOURCE = "outcome_activity_stationary"


def update_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if fieldnames is None or SOURCE not in fieldnames:
            raise ValueError(f"{path} does not contain {SOURCE}")
        rows = list(reader)

    for row in rows:
        stationary = float(row[SOURCE])
        if not 0 <= stationary <= 1:
            raise ValueError(f"{path}: invalid stationary proportion {stationary}")
        row[TARGET] = f"{1 - stationary:.6f}"

    if TARGET not in fieldnames:
        fieldnames.insert(fieldnames.index(SOURCE) + 1, TARGET)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {len(rows):,} rows in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        default=[
            Path("Data/all_combined_data.csv"),
            Path("Data/exploration_data.csv"),
            Path("Data/holdout_data.csv"),
        ],
        help="CSV files to update; defaults to the three project datasets.",
    )
    args = parser.parse_args()
    for path in args.csv_files:
        update_csv(path)


if __name__ == "__main__":
    main()
