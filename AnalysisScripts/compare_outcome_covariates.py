#!/usr/bin/env python3
"""Compare every covariate against outcome_activity_not_stationary.

Uses Data/exploration_data.csv only (holdout stays untouched). Produces:
  - Analysis/outcome_vs_covariates/numeric_correlations.csv
  - Analysis/outcome_vs_covariates/categorical_associations.csv
  - Analysis/outcome_vs_covariates/correlation_bar.png
  - Analysis/outcome_vs_covariates/scatter_top_numeric.png
  - Analysis/outcome_vs_covariates/scatter_bottom_numeric.png
  - Analysis/outcome_vs_covariates/boxplots_categorical.png
  - Analysis/outcome_vs_covariates/top_covariate_correlation_heatmap.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path("Data/exploration_data.csv")
OUT_DIR = Path("Analysis/outcome_vs_covariates")
OUTCOME = "outcome_activity_not_stationary"

NON_COVARIATE_COLUMNS = {
    "observation_id",
    "block_start_timestamp_local",
    "recording_month",
    "block_video_seconds",
    "pre_video_seconds",
    "outcome_video_seconds",
    "weather_hour_start_local",
    OUTCOME,
}
OUTCOME_LEAKAGE_COLUMNS = {
    "outcome_activity_eating",
    "outcome_activity_drinking",
    "outcome_activity_walking",
    "outcome_activity_stationary",
    "outcome_activity_other",
    "outcome_activity_unknown",
}
CATEGORICAL_COLUMNS = ["pre_posture_dominant", "weather_is_day", "recording_month"]

N_TOP = 6


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def numeric_covariates(df: pd.DataFrame) -> list[str]:
    exclude = NON_COVARIATE_COLUMNS | OUTCOME_LEAKAGE_COLUMNS | set(CATEGORICAL_COLUMNS)
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def compute_numeric_correlations(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    constant_cols = []
    for col in cols:
        if df[col].std() == 0:
            constant_cols.append(col)
            continue
        r, p = stats.pearsonr(df[col], df[OUTCOME])
        rows.append({"covariate": col, "pearson_r": r, "p_value": p, "abs_r": abs(r)})
    out = pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
    return out, constant_cols


def compute_categorical_associations(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        groups = [g[OUTCOME].values for _, g in df.groupby(col)]
        f_stat, p = stats.f_oneway(*groups)
        grand_mean = df[OUTCOME].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = ((df[OUTCOME] - grand_mean) ** 2).sum()
        eta_sq = ss_between / ss_total
        rows.append({
            "covariate": col,
            "n_categories": len(groups),
            "f_stat": f_stat,
            "p_value": p,
            "eta_squared": eta_sq,
        })
    out = pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)
    return out


def plot_correlation_bar(corr_df: pd.DataFrame, path: Path) -> None:
    ordered = corr_df.sort_values("pearson_r")
    colors = ["#c0392b" if r < 0 else "#2471a3" for r in ordered["pearson_r"]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(ordered))))
    ax.barh(ordered["covariate"], ordered["pearson_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with outcome_activity_not_stationary")
    ax.set_title("Numeric covariate correlation with outcome")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter_grid(df: pd.DataFrame, cols: list[str], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4), squeeze=False)
    for ax, col in zip(axes[0], cols):
        ax.scatter(df[col], df[OUTCOME], alpha=0.15, s=12, color="#2471a3")
        ax.set_xlabel(col)
        ax.set_ylabel(OUTCOME)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_categorical_boxplots(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.5), squeeze=False)
    for ax, col in zip(axes[0], cols):
        order = sorted(df[col].dropna().unique(), key=lambda x: str(x))
        data = [df.loc[df[col] == level, OUTCOME].values for level in order]
        ax.boxplot(data, tick_labels=[str(level) for level in order], showmeans=True)
        ax.set_title(col)
        ax.set_ylabel(OUTCOME)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_covariate_heatmap(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    corr = df[cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2, 1.1 * len(cols) + 2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pairwise correlation among top covariates")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    num_cols = numeric_covariates(df)
    numeric_corr, constant_cols = compute_numeric_correlations(df, num_cols)
    numeric_corr.to_csv(OUT_DIR / "numeric_correlations.csv", index=False)

    cat_assoc = compute_categorical_associations(df, CATEGORICAL_COLUMNS)
    cat_assoc.to_csv(OUT_DIR / "categorical_associations.csv", index=False)

    plot_correlation_bar(numeric_corr, OUT_DIR / "correlation_bar.png")

    top_numeric = numeric_corr.head(3)["covariate"].tolist()
    bottom_numeric = numeric_corr.tail(3)["covariate"].tolist()
    plot_scatter_grid(df, top_numeric, OUT_DIR / "scatter_top_numeric.png",
                       "Strongest numeric correlates")
    plot_scatter_grid(df, bottom_numeric, OUT_DIR / "scatter_bottom_numeric.png",
                       "Weakest numeric correlates")

    plot_categorical_boxplots(df, CATEGORICAL_COLUMNS, OUT_DIR / "boxplots_categorical.png")

    top_overall_numeric = numeric_corr.head(N_TOP)["covariate"].tolist()
    plot_top_covariate_heatmap(df, top_overall_numeric,
                                OUT_DIR / "top_covariate_correlation_heatmap.png")

    top_overall_numeric_corr = df[top_overall_numeric].corr(method="pearson")
    top_overall_numeric_corr.to_csv(OUT_DIR / "top_covariate_correlation_matrix.csv")

    summary = {
        "n_rows": len(df),
        "numeric_covariates_checked": len(num_cols),
        "constant_numeric_covariates_excluded": constant_cols,
        "categorical_covariates_checked": CATEGORICAL_COLUMNS,
        "strongest_numeric": numeric_corr.head(5).to_dict(orient="records"),
        "weakest_numeric": numeric_corr.tail(5).to_dict(orient="records"),
        "categorical_associations": cat_assoc.to_dict(orient="records"),
        "top_numeric_for_pairwise_check": top_overall_numeric,
    }
    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Rows analyzed: {len(df)}")
    if constant_cols:
        print(f"\nExcluded constant (zero-variance) covariates: {constant_cols}")
    print("\nStrongest numeric correlations with outcome:")
    print(numeric_corr.head(5).to_string(index=False))
    print("\nWeakest numeric correlations with outcome:")
    print(numeric_corr.tail(5).to_string(index=False))
    print("\nCategorical associations (eta-squared):")
    print(cat_assoc.to_string(index=False))
    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
