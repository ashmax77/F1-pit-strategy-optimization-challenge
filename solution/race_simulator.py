#!/usr/bin/env python3
import json
import sys
from pathlib import Path


TIRES = ("SOFT", "MEDIUM", "HARD")
TRACKS = ("Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka")
REF_TEMP = 30.0


def _default_model():
    return {
        "feature_weights": {},
        "feature_scales": {
            "laps": 100.0,
            "ages": 5000.0,
            "ages2": 200000.0,
            "lapidx": 3500.0,
            "lapidx2": 250000.0,
            "stops": 6.0,
            "pit_time_stops": 180.0,
            "max_stint": 70.0,
            "temp_norm": 15.0,
        },
        "metadata": {"source": "default"},
    }


def _load_model():
    model_path = Path(__file__).with_name("model_params.json")
    if not model_path.exists():
        return _default_model()
    with model_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _compound_stats(starting_tire, pit_stops, total_laps):
    laps_by_tire = {t: 0 for t in TIRES}
    age_sum_by_tire = {t: 0.0 for t in TIRES}
    age2_sum_by_tire = {t: 0.0 for t in TIRES}
    lapidx_sum_by_tire = {t: 0.0 for t in TIRES}
    lapidx2_sum_by_tire = {t: 0.0 for t in TIRES}
    stints_by_tire = {t: 0 for t in TIRES}
    max_stint_by_tire = {t: 0 for t in TIRES}

    current_tire = starting_tire
    prev_lap = 0

    for stop in pit_stops:
        stop_lap = int(stop["lap"])
        stint_len = stop_lap - prev_lap
        if stint_len > 0:
            start_lap = prev_lap + 1
            end_lap = stop_lap
            laps_by_tire[current_tire] += stint_len
            age_sum_by_tire[current_tire] += (stint_len * (stint_len + 1)) / 2.0
            age2_sum_by_tire[current_tire] += (
                stint_len * (stint_len + 1) * (2 * stint_len + 1)
            ) / 6.0
            lapidx_sum_by_tire[current_tire] += (
                (start_lap + end_lap) * stint_len
            ) / 2.0
            lapidx2_sum_by_tire[current_tire] += (
                end_lap * (end_lap + 1) * (2 * end_lap + 1)
                - (start_lap - 1) * start_lap * (2 * start_lap - 1)
            ) / 6.0
            stints_by_tire[current_tire] += 1
            max_stint_by_tire[current_tire] = max(max_stint_by_tire[current_tire], stint_len)
        current_tire = stop["to_tire"]
        prev_lap = stop_lap

    final_stint_len = total_laps - prev_lap
    if final_stint_len > 0:
        start_lap = prev_lap + 1
        end_lap = total_laps
        laps_by_tire[current_tire] += final_stint_len
        age_sum_by_tire[current_tire] += (final_stint_len * (final_stint_len + 1)) / 2.0
        age2_sum_by_tire[current_tire] += (
            final_stint_len * (final_stint_len + 1) * (2 * final_stint_len + 1)
        ) / 6.0
        lapidx_sum_by_tire[current_tire] += (
            (start_lap + end_lap) * final_stint_len
        ) / 2.0
        lapidx2_sum_by_tire[current_tire] += (
            end_lap * (end_lap + 1) * (2 * end_lap + 1)
            - (start_lap - 1) * start_lap * (2 * start_lap - 1)
        ) / 6.0
        stints_by_tire[current_tire] += 1
        max_stint_by_tire[current_tire] = max(max_stint_by_tire[current_tire], final_stint_len)

    return (
        laps_by_tire,
        age_sum_by_tire,
        age2_sum_by_tire,
        lapidx_sum_by_tire,
        lapidx2_sum_by_tire,
        stints_by_tire,
        max_stint_by_tire,
    )


def _extract_features(strategy, race_config, model):
    scales = model["feature_scales"]

    total_laps = int(race_config["total_laps"])
    track = race_config["track"]
    track_temp = float(race_config["track_temp"])
    temp_norm = (track_temp - REF_TEMP) / scales["temp_norm"]
    pit_lane_time = float(race_config["pit_lane_time"])
    pit_stops = strategy.get("pit_stops", [])

    (
        laps_by_tire,
        age_sum_by_tire,
        age2_sum_by_tire,
        lapidx_sum_by_tire,
        lapidx2_sum_by_tire,
        stints_by_tire,
        max_stint_by_tire,
    ) = _compound_stats(strategy["starting_tire"], pit_stops, total_laps)

    feats = {
        "stops": len(pit_stops) / scales["stops"],
        "pit_time_stops": (pit_lane_time * len(pit_stops)) / scales["pit_time_stops"],
        f"drv_{strategy['driver_id']}": 1.0,
    }

    for tire in TIRES:
        key = tire.lower()
        laps = laps_by_tire[tire] / scales["laps"]
        ages = age_sum_by_tire[tire] / scales["ages"]
        ages2 = age2_sum_by_tire[tire] / scales["ages2"]
        lapidx = lapidx_sum_by_tire[tire] / scales["lapidx"]
        lapidx2 = lapidx2_sum_by_tire[tire] / scales["lapidx2"]
        stints = stints_by_tire[tire] / scales["stops"]
        max_stint = max_stint_by_tire[tire] / scales["max_stint"]

        feats[f"laps_{key}"] = laps
        feats[f"ages_{key}"] = ages
        feats[f"ages2_{key}"] = ages2
        feats[f"lapidx_{key}"] = lapidx
        feats[f"lapidx2_{key}"] = lapidx2
        feats[f"stints_{key}"] = stints
        feats[f"maxstint_{key}"] = max_stint

        feats[f"temp_laps_{key}"] = temp_norm * laps
        feats[f"temp_ages_{key}"] = temp_norm * ages
        feats[f"temp_ages2_{key}"] = temp_norm * ages2
        feats[f"temp_lapidx_{key}"] = temp_norm * lapidx
        feats[f"temp_lapidx2_{key}"] = temp_norm * lapidx2

        if track in TRACKS:
            feats[f"trk_{track}_laps_{key}"] = laps
            feats[f"trk_{track}_ages_{key}"] = ages
            feats[f"trk_{track}_lapidx_{key}"] = lapidx

    return feats


def _driver_score(strategy, race_config, model):
    weights = model["feature_weights"]
    feats = _extract_features(strategy, race_config, model)

    # Base lap time is identical for all drivers in a race, so we can rank with a relative score.
    score = 0.0
    for key, value in feats.items():
        w = weights.get(key)
        if w is not None:
            score += w * value
    return score


def simulate_race(race_config, strategies, model):
    scored = []
    for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
        strategy = strategies[pos_key]
        driver_id = strategy["driver_id"]
        score = _driver_score(strategy, race_config, model)
        scored.append((score, driver_id))

    scored.sort(key=lambda x: x[0])
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