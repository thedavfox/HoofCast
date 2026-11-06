#!/usr/bin/env python3
"""Randomly split AllData.csv into train/validate/test CSVs by row."""

import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Full observation CSV (e.g. Data/AllData.csv)")
    parser.add_argument("--outdir", type=Path, default=Path("Data"))
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--validate-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=226)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    if not args.yes:
        answer = input(
            "This will (re)generate train_data.csv, validate_data.csv, and test_data.csv, "
            "overwriting any existing versions in the output directory. Any models or results "
            "already trained/evaluated on the current splits may no longer match. "
            "Are you sure you want to proceed? [y/N]: "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted: no files were written.")
            return

    with args.source.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames, rows = reader.fieldnames, list(reader)

    random.Random(args.seed).shuffle(rows)

    n = len(rows)
    n_train = round(n * args.train_frac)
    n_validate = round(n * args.validate_frac)
    splits = {
        "train_data": rows[:n_train],
        "validate_data": rows[n_train:n_train + n_validate],
        "test_data": rows[n_train + n_validate:],
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        out_path = args.outdir / f"{name}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"Wrote {len(split_rows):,} rows to {out_path}")


if __name__ == "__main__":
    main()
