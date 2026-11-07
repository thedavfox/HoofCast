#!/usr/bin/env python3
"""Add pre_posture_dominant to the project observation CSVs.

The derived categorical is the posture with the largest fraction of time
in the pre-window: argmax over the pre_posture_* proportion columns.
"""

import argparse
import csv
from pathlib import Path


TARGET = "pre_posture_dominant"
SOURCE_COLUMNS = [
    "pre_posture_standing",
    "pre_posture_lying_sternal",
    "pre_posture_lying_lateral",
    "pre_posture_transitioning",
    "pre_posture_unknown",
]


def update_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if fieldnames is None or not all(col in fieldnames for col in SOURCE_COLUMNS):
            raise ValueError(f"{path} is missing one or more pre_posture_* columns")
        rows = list(reader)

    for row in rows:
        fractions = {col: float(row[col]) for col in SOURCE_COLUMNS}
        dominant = max(fractions, key=fractions.get)
        row[TARGET] = dominant.removeprefix("pre_posture_")

    if TARGET not in fieldnames:
        fieldnames.insert(fieldnames.index(SOURCE_COLUMNS[-1]) + 1, TARGET)
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
