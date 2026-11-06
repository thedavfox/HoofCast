#!/usr/bin/env python3
"""Add hourly Open-Meteo ERA5 weather covariates to observation blocks."""

import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo


LATITUDE, LONGITUDE = 47.6262, -122.5212  # Bainbridge Island, WA
TIMEZONE, MODEL = "America/Los_Angeles", "era5"
WEATHER = ["temperature_2m", "relative_humidity_2m", "precipitation", "cloud_cover",
           "shortwave_radiation", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "is_day"]
NAMES = {
    "temperature_2m": "weather_temperature_c", "relative_humidity_2m": "weather_relative_humidity_pct",
    "precipitation": "weather_precipitation_mm_prior_hour", "cloud_cover": "weather_cloud_cover_pct",
    "shortwave_radiation": "weather_shortwave_radiation_w_m2_prior_hour",
    "wind_speed_10m": "weather_wind_speed_10m_kmh", "wind_direction_10m": "weather_wind_direction_10m_degrees",
    "wind_gusts_10m": "weather_wind_gusts_10m_kmh", "is_day": "weather_is_day",
}


def fetch_weather(start_date, end_date):
    params = {"latitude": LATITUDE, "longitude": LONGITUDE, "start_date": start_date, "end_date": end_date,
              "hourly": ",".join(WEATHER), "timezone": TIMEZONE, "timeformat": "unixtime",
              "models": MODEL, "cell_selection": "land"}
    url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
    response = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", url],
                              check=True, capture_output=True, text=True, timeout=60)
    payload = json.loads(response.stdout)
    zone, hourly = ZoneInfo(TIMEZONE), payload["hourly"]
    values = {}
    for index, epoch in enumerate(hourly["time"]):
        local = dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).astimezone(zone).replace(tzinfo=None)
        values[local.strftime("%Y-%m-%d %H:00:00")] = {field: hourly[field][index] for field in WEATHER}
    return values, url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="200-second observation CSV")
    parser.add_argument("output", type=Path, help="Weather-enriched CSV")
    parser.add_argument("--provenance", type=Path, help="Optional Markdown source note")
    args = parser.parse_args()
    with args.source.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns, rows = reader.fieldnames or [], list(reader)
    timestamps = [dt.datetime.fromisoformat(row["block_start_timestamp_local"]) for row in rows]
    weather, url = fetch_weather(str(min(timestamps).date()), str(max(timestamps).date()))
    for row, timestamp in zip(rows, timestamps):
        key = timestamp.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:00:00")
        if key not in weather:
            raise ValueError(f"No weather data available for {key}")
        row["weather_hour_start_local"] = key
        row.update({NAMES[field]: weather[key][field] for field in WEATHER})
    fieldnames = columns + ["weather_hour_start_local"] + list(NAMES.values())
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.provenance:
        args.provenance.write_text(
            "# Weather data provenance\n\n"
            "Open-Meteo Historical Weather API, ERA5 model, Bainbridge Island, Washington "
            f"({LATITUDE}, {LONGITUDE}), land grid-cell selection. Local matching timezone: {TIMEZONE}.\n\n"
            f"API request: {url}\n\n"
            "Each observation receives weather from the hour containing its block-start timestamp. "
            "Precipitation is the preceding-hour total; shortwave radiation is the preceding-hour mean. "
            "ERA5 is gridded reanalysis data, not an on-site weather-station measurement.\n",
            encoding="utf-8")
    print(f"Wrote {len(rows):,} weather-enriched observations to {args.output}")


if __name__ == "__main__":
    main()
