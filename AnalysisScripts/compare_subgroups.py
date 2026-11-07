#!/usr/bin/env python3
"""Compare covariate means across natural subgroups.

Uses Data/exploration_data.csv. Subgroups: pre_posture_dominant,
weather_is_day, recording_month. For each, computes group means (and
group sizes) of every numeric covariate plus the outcome, and a
one-way ANOVA F-stat / eta-squared per covariate to flag which
covariates differ most across each grouping.

Outputs go to Analysis/subgroup_comparison/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

DATA_PATH = Path("Data/exploration_data.csv")
OUT_DIR = Path("Analysis/subgroup_comparison")
OUTCOME = "outcome_activity_not_stationary"

GROUP_COLUMNS = ["pre_posture_dominant", "weather_is_day", "recording_month"]

NON_NUMERIC_OR_ID = {
    "observation_id",
    "block_start_timestamp_local",
    "recording_month",
    "block_video_seconds",
    "pre_video_seconds",
    "outcome_video_seconds",
    "weather_hour_start_local",
    "weather_is_day",
}


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in NON_NUMERIC_OR_ID or col == "pre_posture_dominant":
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].std() > 0:
            cols.append(col)
    return cols


def group_means_table(df: pd.DataFrame, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    means = df.groupby(group_col)[value_cols].mean()
    counts = df.groupby(group_col).size().rename("n")
    table = counts.to_frame().join(means)
    return table.reset_index()


def group_difference_strength(df: pd.DataFrame, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in value_cols:
        groups = [g[col].values for _, g in df.groupby(group_col)]
        if len(groups) < 2:
            continue
        f_stat, p = stats.f_oneway(*groups)
        grand_mean = df[col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((df[col] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total if ss_total > 0 else float("nan")
        rows.append({"covariate": col, "f_stat": f_stat, "p_value": p, "eta_squared": eta_sq})
    out = pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)
    return out


def plot_group_means(table: pd.DataFrame, group_col: str, top_cols: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(top_cols), figsize=(4 * len(top_cols), 4), squeeze=False)
    for ax, col in zip(axes[0], top_cols):
        ax.bar(table[group_col].astype(str), table[col], color="#2471a3")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Group means by {group_col}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    value_cols = numeric_columns(df) + [OUTCOME]
    value_cols = list(dict.fromkeys(value_cols))

    all_summaries = {}
    for group_col in GROUP_COLUMNS:
        means_table = group_means_table(df, group_col, value_cols)
        means_table.to_csv(OUT_DIR / f"group_means_{group_col}.csv", index=False)

        diff_strength = group_difference_strength(df, group_col, value_cols)
        diff_strength.to_csv(OUT_DIR / f"group_differences_{group_col}.csv", index=False)

        top_cols = diff_strength.head(4)["covariate"].tolist()
        plot_group_means(means_table, group_col, top_cols, OUT_DIR / f"group_means_{group_col}.png")

        all_summaries[group_col] = {
            "group_sizes": means_table[[group_col, "n"]].to_dict(orient="records"),
            "top_differing_covariates": diff_strength.head(5).to_dict(orient="records"),
        }

        print(f"\n=== Subgroup: {group_col} ===")
        print(means_table[[group_col, "n"]].to_string(index=False))
        print("\nTop differing covariates (eta-squared):")
        print(diff_strength.head(5).to_string(index=False))

    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
