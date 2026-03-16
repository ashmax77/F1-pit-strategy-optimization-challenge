#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import race_simulator


def base_scored(case_data, model):
    race_config = case_data["race_config"]
    strategies = case_data["strategies"]
    rows = []
    for pos_key in sorted(strategies.keys(), key=lambda key: int(key[3:])):
        strategy = strategies[pos_key]
        driver_id = strategy["driver_id"]
        if "feature_weights" in model and "mechanistic_params" not in model:
            score = race_simulator._driver_relative_time_legacy(strategy, race_config, model)
        else:
            score = race_simulator._driver_relative_time(strategy, race_config, model)
        rows.append((score, driver_id, strategy))
    rows.sort(key=lambda item: (item[0], item[1]))
    return rows


def main():
    repo_root = Path(__file__).resolve().parents[1]
    model = json.loads((repo_root / "solution" / "model_params.json").read_text(encoding="utf-8"))
    inputs_dir = repo_root / "data" / "test_cases" / "inputs"
    expected_dir = repo_root / "data" / "test_cases" / "expected_outputs"

    close_inversion_tracks = Counter()
    close_inversion_patterns = Counter()
    gap_bands = Counter()
    same_strategy_inversions = 0
    total_failed = 0

    for i in range(1, 101):
        name = f"test_{i:03d}.json"
        case_data = json.loads((inputs_dir / name).read_text(encoding="utf-8"))
        expected = json.loads((expected_dir / name).read_text(encoding="utf-8"))["finishing_positions"]
        predicted = race_simulator.simulate_race(case_data["race_config"], case_data["strategies"], model)
        if predicted == expected:
            continue

        total_failed += 1
        scored = base_scored(case_data, model)
        pred_index = {driver_id: idx for idx, driver_id in enumerate([driver_id for _score, driver_id, _strategy in scored])}
        by_driver = {strategy["driver_id"]: strategy for strategy in case_data["strategies"].values()}

        for left in range(len(expected) - 1):
            fast = expected[left]
            slow = expected[left + 1]
            if pred_index[fast] < pred_index[slow]:
                continue

            fast_row = next(row for row in scored if row[1] == fast)
            slow_row = next(row for row in scored if row[1] == slow)
            gap = abs(slow_row[0] - fast_row[0])
            if gap > 2.5:
                continue

            track = case_data["race_config"]["track"]
            close_inversion_tracks[track] += 1

            fast_strategy = by_driver[fast]
            slow_strategy = by_driver[slow]
            fast_last_tire = fast_strategy.get("pit_stops", [])[-1]["to_tire"] if fast_strategy.get("pit_stops") else fast_strategy["starting_tire"]
            slow_last_tire = slow_strategy.get("pit_stops", [])[-1]["to_tire"] if slow_strategy.get("pit_stops") else slow_strategy["starting_tire"]
            temp = float(case_data["race_config"]["track_temp"])
            regime = "hot" if temp >= 33 else ("cold" if temp <= 27 else "mid")
            close_inversion_patterns[(track, regime, fast_last_tire, slow_last_tire)] += 1

            band = "<=0.25" if gap <= 0.25 else ("<=0.75" if gap <= 0.75 else "<=1.5" if gap <= 1.5 else "<=2.5")
            gap_bands[band] += 1

            if race_simulator._strategy_signature(fast_strategy) == race_simulator._strategy_signature(slow_strategy):
                same_strategy_inversions += 1

    output = {
        "failed_races": total_failed,
        "close_inversion_tracks": close_inversion_tracks.most_common(10),
        "close_inversion_patterns": [
            {"track": track, "regime": regime, "expected_fast_final_tire": fast_tire, "expected_slow_final_tire": slow_tire, "count": count}
            for (track, regime, fast_tire, slow_tire), count in close_inversion_patterns.most_common(15)
        ],
        "gap_bands": dict(gap_bands),
        "same_strategy_inversions": same_strategy_inversions,
    }

    out_path = repo_root / "solution" / "rerank_diagnostics_visible.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()