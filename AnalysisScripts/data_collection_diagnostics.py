#!/usr/bin/env python3
"""Summarize data-collection coverage and quality issues.

Covers: recording span and monthly/hourly coverage, visibility quality
(how often the horse was fully/partially/not visible), rare or absent
behavior labels, class balance of the outcome, and block-duration
consistency. Uses Data/all_combined_data.csv (full pool, pre-split).

Outputs go to Analysis/data_collection_diagnostics/.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("Data/all_combined_data.csv")
OUT_DIR = Path("Analysis/data_collection_diagnostics")
OUTCOME = "outcome_activity_not_stationary"

SECONDARY_BEHAVIOR_COLS = [
    "pre_pawing",
    "pre_biting_object",
    "pre_rubbing",
    "pre_repetitive_movement",
    "pre_no_secondary_behavior",
]
POSTURE_COLS = [
    "pre_posture_standing",
    "pre_posture_lying_sternal",
    "pre_posture_lying_lateral",
    "pre_posture_transitioning",
    "pre_posture_unknown",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["block_start_timestamp_local"] = pd.to_datetime(df["block_start_timestamp_local"], format="mixed")

    coverage = {
        "n_observations": len(df),
        "date_range_start": str(df["block_start_timestamp_local"].min()),
        "date_range_end": str(df["block_start_timestamp_local"].max()),
        "months_covered": df["recording_month"].value_counts().to_dict(),
        "block_video_seconds_unique_values": df["block_video_seconds"].unique().tolist(),
        "pre_video_seconds_unique_values": df["pre_video_seconds"].unique().tolist(),
        "outcome_video_seconds_unique_values": df["outcome_video_seconds"].unique().tolist(),
    }

    hour_counts = df["time_of_day_hour"].value_counts().sort_index()
    hour_counts.to_csv(OUT_DIR / "hourly_coverage.csv", header=["n_observations"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(hour_counts.index, hour_counts.values, color="#2471a3")
    ax.set_xlabel("time_of_day_hour")
    ax.set_ylabel("n_observations")
    ax.set_title("Observation coverage by hour of day")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hourly_coverage.png", dpi=150)
    plt.close(fig)

    visibility_summary = df[["pre_visibility_full", "pre_visibility_partial",
                              "pre_visibility_not_visible"]].mean().to_dict()
    pct_mostly_not_visible = float((df["pre_visibility_not_visible"] > 0.5).mean())
    pct_any_not_visible = float((df["pre_visibility_not_visible"] > 0).mean())

    behavior_rates = {}
    for col in SECONDARY_BEHAVIOR_COLS:
        behavior_rates[col] = {
            "mean_proportion": float(df[col].mean()),
            "pct_observations_with_any": float((df[col] > 0).mean()),
            "is_constant_zero": bool(df[col].std() == 0),
        }

    posture_rates = {col: float(df[col].mean()) for col in POSTURE_COLS}
    pct_posture_unknown_any = float((df["pre_posture_unknown"] > 0).mean())

    outcome_summary = {
        "mean": float(df[OUTCOME].mean()),
        "median": float(df[OUTCOME].median()),
        "std": float(df[OUTCOME].std()),
        "pct_exactly_0": float((df[OUTCOME] == 0).mean()),
        "pct_exactly_1": float((df[OUTCOME] == 1).mean()),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[OUTCOME], bins=20, color="#2471a3", edgecolor="white")
    ax.set_xlabel(OUTCOME)
    ax.set_ylabel("count")
    ax.set_title("Outcome distribution (class balance check)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "outcome_distribution.png", dpi=150)
    plt.close(fig)

    dup_ids = int(df["observation_id"].duplicated().sum())
    gap_detection = df.sort_values("block_start_timestamp_local")["block_start_timestamp_local"].diff()
    median_gap_seconds = gap_detection.dt.total_seconds().median()

    summary = {
        "coverage": coverage,
        "duplicate_observation_ids": dup_ids,
        "median_gap_between_consecutive_block_starts_seconds": median_gap_seconds,
        "visibility_mean_proportions": visibility_summary,
        "pct_observations_more_than_half_not_visible": pct_mostly_not_visible,
        "pct_observations_with_any_not_visible_time": pct_any_not_visible,
        "posture_mean_proportions": posture_rates,
        "pct_observations_with_any_unknown_posture_time": pct_posture_unknown_any,
        "secondary_behavior_rates": behavior_rates,
        "outcome_class_balance": outcome_summary,
    }
    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"n_observations: {len(df)}")
    print(f"date range: {coverage['date_range_start']} to {coverage['date_range_end']}")
    print(f"months covered: {coverage['months_covered']}")
    print(f"duplicate observation_ids: {dup_ids}")
    print(f"median gap between consecutive block starts: {median_gap_seconds:.1f} sec")
    print("\nVisibility mean proportions:", visibility_summary)
    print(f"Observations with >50% not-visible time: {pct_mostly_not_visible:.2%}")
    print(f"Observations with any not-visible time: {pct_any_not_visible:.2%}")
    print("\nSecondary behavior rates (mean proportion, % with any occurrence):")
    for col, stats in behavior_rates.items():
        print(f"  {col}: mean={stats['mean_proportion']:.4f}, "
              f"any={stats['pct_observations_with_any']:.2%}, "
              f"constant_zero={stats['is_constant_zero']}")
    print("\nOutcome class balance:", outcome_summary)
    print(f"\nOutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
