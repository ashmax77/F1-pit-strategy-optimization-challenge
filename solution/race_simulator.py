#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REF_TEMP = 30.0


def _default_model():
    return {
        "feature_weights": {},
        "mechanistic_config": {
            "age_bucket_cap": 40,
            "progress_buckets": 8,
            "temp_scale": 15.0,
        },
        "metadata": {"source": "default"},
    }


def _load_model():
    model_path = Path(__file__).with_name("model_params.json")
    if not model_path.exists():
        return _default_model()
    with model_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _age_bucket(age, age_bucket_cap):
    return min(age, age_bucket_cap)


def _phase_bucket(lap_number, total_laps, progress_buckets):
    return min(progress_buckets - 1, ((lap_number - 1) * progress_buckets) // total_laps)


def _driver_relative_time(strategy, race_config, model):
    weights = model.get("feature_weights", {})
    config = model.get("mechanistic_config", {})
    age_bucket_cap = int(config.get("age_bucket_cap", 40))
    progress_buckets = int(config.get("progress_buckets", 8))
    temp_scale = float(config.get("temp_scale", 15.0))

    total_laps = int(race_config["total_laps"])
    track = race_config["track"]
    track_temp = float(race_config["track_temp"])
    pit_lane_time = float(race_config["pit_lane_time"])
    pit_stops = strategy.get("pit_stops", [])
    stop_map = {int(stop["lap"]): stop["to_tire"] for stop in pit_stops}
    temp_norm = (track_temp - REF_TEMP) / temp_scale

    relative_time = 0.0
    relative_time += weights.get(f"driver::{strategy['driver_id']}", 0.0)
    relative_time += weights.get("pit_count", 0.0) * len(pit_stops)
    relative_time += weights.get("pit_lane_time", 0.0) * (pit_lane_time * len(pit_stops))
    relative_time += weights.get(f"track_pit_count::{track}", 0.0) * len(pit_stops)
    relative_time += weights.get(f"track_pit_time::{track}", 0.0) * (pit_lane_time * len(pit_stops))

    for stop in pit_stops:
        stop_lap = int(stop["lap"])
        from_tire = stop["from_tire"]
        to_tire = stop["to_tire"]
        stop_phase = _phase_bucket(stop_lap, total_laps, progress_buckets)
        relative_time += weights.get(f"pit_trans::{from_tire}->{to_tire}", 0.0)
        relative_time += weights.get(f"pit_trans_phase::{from_tire}->{to_tire}::{stop_phase}", 0.0)
        relative_time += weights.get(f"track_pit_trans::{track}::{from_tire}->{to_tire}", 0.0)

    current_tire = strategy["starting_tire"]
    tire_age = 0

    for lap_number in range(1, total_laps + 1):
        tire_age += 1
        age_bucket = _age_bucket(tire_age, age_bucket_cap)
        phase_bucket = _phase_bucket(lap_number, total_laps, progress_buckets)

        relative_time += weights.get(f"lap::{current_tire}::{age_bucket}", 0.0)
        relative_time += weights.get(f"temp::{current_tire}::{age_bucket}", 0.0) * temp_norm
        relative_time += weights.get(f"phase::{current_tire}::{phase_bucket}", 0.0)
        relative_time += weights.get(f"track_lap::{track}::{current_tire}::{age_bucket}", 0.0)
        relative_time += weights.get(f"track_phase::{track}::{current_tire}::{phase_bucket}", 0.0)

        if lap_number in stop_map:
            current_tire = stop_map[lap_number]
            tire_age = 0

    return relative_time


def simulate_race(race_config, strategies, model):
    scored = []
    for pos_key in sorted(strategies.keys(), key=lambda key: int(key[3:])):
        strategy = strategies[pos_key]
        driver_id = strategy["driver_id"]
        relative_time = _driver_relative_time(strategy, race_config, model)
        scored.append((relative_time, driver_id))

    scored.sort(key=lambda item: item[0])
    return [driver_id for _, driver_id in scored]


def main():
    test_case = json.load(sys.stdin)
    race_id = test_case["race_id"]
    race_config = test_case["race_config"]
    strategies = test_case["strategies"]

    model = _load_model()
    finishing_positions = simulate_race(race_config, strategies, model)

    output = {
        "race_id": race_id,
        "finishing_positions": finishing_positions,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()