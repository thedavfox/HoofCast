#!/usr/bin/env python3
"""Check for missing (NA/NULL/blank) values across the project CSVs.

Produces Analysis/missing_values/missing_value_report.csv and
Analysis/missing_values/summary.json, covering all_combined_data.csv,
exploration_data.csv, and holdout_data.csv.
"""

import json
from pathlib import Path

import pandas as pd

DATA_FILES = {
    "all_combined_data": Path("Data/all_combined_data.csv"),
    "exploration_data": Path("Data/exploration_data.csv"),
    "holdout_data": Path("Data/holdout_data.csv"),
}
OUT_DIR = Path("Analysis/missing_values")


def check_file(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=True, na_values=["", "NA", "NULL", "null", "NaN", "nan"])
    n_rows = len(df)
    rows = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if df[col].dtype == object:
            n_blank_string = (df[col].astype(str).str.strip() == "").sum()
        else:
            n_blank_string = 0
        rows.append({
            "dataset": name,
            "column": col,
            "n_rows": n_rows,
            "n_missing": int(n_missing),
            "pct_missing": round(100 * n_missing / n_rows, 4),
            "n_blank_string": int(n_blank_string),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_reports = []
    for name, path in DATA_FILES.items():
        all_reports.append(check_file(name, path))
    report = pd.concat(all_reports, ignore_index=True)
    report.to_csv(OUT_DIR / "missing_value_report.csv", index=False)

    nonzero = report[(report["n_missing"] > 0) | (report["n_blank_string"] > 0)]
    summary = {
        "datasets_checked": list(DATA_FILES.keys()),
        "total_columns_checked_per_dataset": int(report.groupby("dataset").size().iloc[0]),
        "columns_with_any_missing": nonzero.to_dict(orient="records"),
        "any_missing_values_found": bool(len(nonzero) > 0),
    }
    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Checked {len(DATA_FILES)} datasets, {summary['total_columns_checked_per_dataset']} columns each.")
    if len(nonzero) == 0:
        print("No missing (NA/NULL/blank) values found in any dataset.")
    else:
        print("Columns with missing values:")
        print(nonzero.to_string(index=False))
    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
