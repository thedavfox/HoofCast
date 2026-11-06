#!/usr/bin/env python3
"""Randomly split AllData.csv into exploration/holdout CSVs by row."""

import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Full observation CSV (e.g. Data/AllData.csv)")
    parser.add_argument("--outdir", type=Path, default=Path("Data"))
    parser.add_argument("--exploration-frac", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=226)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = parser.parse_args()

    if not args.yes:
        answer = input(
            "This will (re)generate exploration_data.csv and holdout_data.csv, "
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
    n_exploration = round(n * args.exploration_frac)
    splits = {
        "exploration_data": rows[:n_exploration],
        "holdout_data": rows[n_exploration:],
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
