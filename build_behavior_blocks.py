#!/usr/bin/env python3
"""Aggregate clip-level horse labels into contiguous 200-second observations.

Each observation uses the first 100 video seconds for pre-period covariates
and the last 100 seconds for activity outcomes.  Blocks never cross a
60-second recording gap or a calendar-month boundary.
"""

import argparse
import csv
import datetime as dt
import json
from pathlib import Path


POSTURES = ["standing", "lying_sternal", "lying_lateral", "transitioning", "unknown"]
VISIBILITIES = ["full", "partial", "not_visible"]
TRANSITIONS = ["stand_to_lie", "lie_to_stand", "none", "unknown"]
ACTIVITIES = ["eating", "drinking", "walking", "stationary", "other", "unknown"]
SECONDARY = ["pawing", "biting_object", "rubbing", "repetitive_movement"]


def add_label(totals, row, seconds, period):
    """Accumulate seconds for one labeled segment into pre or outcome fields."""
    if period == "outcome":
        activity = row.get("activity")
        activity = activity if activity in ACTIVITIES else "unknown"
        totals[f"outcome_activity_{activity}"] += seconds
        return

    posture = row.get("posture")
    visibility = row.get("visibility")
    transition = row.get("transition")
    totals[f"pre_posture_{posture if posture in POSTURES else 'unknown'}"] += seconds
    totals[f"pre_visibility_{visibility if visibility in VISIBILITIES else 'not_visible'}"] += seconds
    totals[f"pre_transition_{transition if transition in TRANSITIONS else 'unknown'}"] += seconds
    try:
        behaviors = set(json.loads(row.get("secondary_behaviors") or "[]"))
    except json.JSONDecodeError:
        behaviors = set()
    for behavior in SECONDARY:
        if behavior in behaviors:
            totals[f"pre_{behavior}"] += seconds
    if not behaviors:
        totals["pre_no_secondary_behavior"] += seconds


def load_sequences(source, max_gap_seconds):
    clips = []
    with source.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row.get("error") or not row.get("recorded_timestamp_local"):
                continue
            duration = float(row.get("duration_seconds") or 0)
            if duration:
                clips.append((dt.datetime.fromisoformat(row["recorded_timestamp_local"]), duration, row))
    clips.sort(key=lambda clip: clip[0])

    sequences, sequence = [], []
    for clip in clips:
        if sequence:
            prior_start, prior_duration, _ = sequence[-1]
            gap = (clip[0] - prior_start - dt.timedelta(seconds=prior_duration)).total_seconds()
            month_changed = (clip[0].year, clip[0].month) != (prior_start.year, prior_start.month)
            if gap > max_gap_seconds or month_changed:
                sequences.append(sequence)
                sequence = []
        sequence.append(clip)
    if sequence:
        sequences.append(sequence)
    return sequences


def build_observations(sequences):
    fields = (
        [f"pre_posture_{x}" for x in POSTURES]
        + [f"pre_visibility_{x}" for x in VISIBILITIES]
        + [f"pre_transition_{x}" for x in TRANSITIONS]
        + [f"pre_{x}" for x in SECONDARY] + ["pre_no_secondary_behavior"]
        + [f"outcome_activity_{x}" for x in ACTIVITIES]
    )
    observations = []
    for sequence in sequences:
        clip_index, offset = 0, 0.0
        while clip_index < len(sequence):
            available = sum(item[1] for item in sequence[clip_index:]) - offset
            if available < 200:
                break
            start, _, _ = sequence[clip_index]
            totals, filled = {field: 0.0 for field in fields}, 0.0
            block_start = start + dt.timedelta(seconds=offset)
            while filled < 200:
                _, duration, row = sequence[clip_index]
                take = min(duration - offset, 200 - filled)
                boundary = 100 - filled
                if 0 < boundary < take:
                    add_label(totals, row, boundary, "pre")
                    add_label(totals, row, take - boundary, "outcome")
                else:
                    add_label(totals, row, take, "pre" if filled < 100 else "outcome")
                filled += take
                offset += take
                if offset >= duration - 1e-8:
                    clip_index, offset = clip_index + 1, 0.0
            observation = {
                "observation_id": f"obs_{len(observations) + 1:05d}",
                "block_start_timestamp_local": block_start.isoformat(sep=" "),
                "time_of_day_hour": block_start.hour,
                "recording_month": block_start.strftime("%Y-%m"),
                "block_video_seconds": 200.0,
                "pre_video_seconds": 100.0,
                "outcome_video_seconds": 100.0,
            }
            observation.update({field: round(totals[field] / 100, 6) for field in fields})
            observations.append(observation)
    return observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Clip-level labeling CSV")
    parser.add_argument("output", type=Path, help="Output observation CSV")
    parser.add_argument("--max-gap-seconds", type=float, default=60)
    args = parser.parse_args()
    observations = build_observations(load_sequences(args.source, args.max_gap_seconds))
    if not observations:
        raise SystemExit("No complete 200-second observations were available.")
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=observations[0].keys())
        writer.writeheader()
        writer.writerows(observations)
    print(f"Wrote {len(observations):,} observations to {args.output}")


if __name__ == "__main__":
    main()
