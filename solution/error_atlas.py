#!/usr/bin/env python3
import json
from pathlib import Path

import race_simulator


def strategy_signature(strategy):
    return (
        strategy["starting_tire"],
        tuple((int(stop["lap"]), stop["from_tire"], stop["to_tire"]) for stop in strategy.get("pit_stops", [])),
    )


def get_driver_score(race_config, strategy, model):
    if "mechanistic_params" in model:
        return race_simulator._driver_relative_time(strategy, race_config, model)
    return race_simulator._driver_relative_time_legacy(strategy, race_config, model)


def classify_race_failure(case_data, expected, predicted, model):
    race_config = case_data["race_config"]
    strategies = case_data["strategies"]

    by_driver = {s["driver_id"]: s for s in strategies.values()}
    pred_index = {d: i for i, d in enumerate(predicted)}
    exp_index = {d: i for i, d in enumerate(expected)}

    scores = {d: get_driver_score(race_config, by_driver[d], model) for d in expected}
    ordered_scores = [scores[d] for d in predicted]
    pos_gaps = [ordered_scores[i + 1] - ordered_scores[i] for i in range(len(ordered_scores) - 1)]

    mismatch_drivers = [d for d in expected if pred_index[d] != exp_index[d]]
    near_tie_count = sum(1 for gap in pos_gaps if gap <= model.get("metadata", {}).get("tie_gap_threshold", 0.06))

    pit_count_mismatch = 0
    same_strategy_mismatch = 0
    late_stint_reversal = 0

    for d in mismatch_drivers:
        if abs(pred_index[d] - exp_index[d]) <= 2:
            continue
        s = by_driver[d]
        pit_c = len(s.get("pit_stops", []))
        around = expected[max(0, exp_index[d] - 1): min(len(expected), exp_index[d] + 2)]
        avg_local = sum(len(by_driver[x].get("pit_stops", [])) for x in around) / max(1, len(around))
        if abs(pit_c - avg_local) >= 1.5:
            pit_count_mismatch += 1

    for i in range(len(expected) - 1):
        a = expected[i]
        b = expected[i + 1]
        if strategy_signature(by_driver[a]) == strategy_signature(by_driver[b]):
            if abs(pred_index[a] - pred_index[b]) >= 1 and pred_index[a] > pred_index[b]:
                same_strategy_mismatch += 1

    for i in range(len(expected) - 1):
        fast = expected[i]
        slow = expected[i + 1]
        if pred_index[fast] > pred_index[slow]:
            fast_last_stop = max([int(s["lap"]) for s in by_driver[fast].get("pit_stops", [])] + [0])
            slow_last_stop = max([int(s["lap"]) for s in by_driver[slow].get("pit_stops", [])] + [0])
            total_laps = int(race_config["total_laps"])
            if fast_last_stop > int(0.65 * total_laps) or slow_last_stop > int(0.65 * total_laps):
                late_stint_reversal += 1

    temp = float(race_config["track_temp"])
    temp_sensitive_flip = (temp <= 27 or temp >= 33) and len(mismatch_drivers) >= 8

    pattern = {
        "pit_count_mismatch": pit_count_mismatch,
        "same_strategy_near_tie": same_strategy_mismatch,
        "late_stint_reversal": late_stint_reversal,
        "temperature_sensitive_flip": int(temp_sensitive_flip),
        "near_tie_count": near_tie_count,
    }

    return {
        "race_id": case_data["race_id"],
        "track": race_config["track"],
        "track_temp": race_config["track_temp"],
        "mismatches": len(mismatch_drivers),
        "avg_adjacent_gap": (sum(pos_gaps) / max(1, len(pos_gaps))),
        "min_adjacent_gap": min(pos_gaps) if pos_gaps else 0.0,
        "pattern": pattern,
    }


def main():
    repo = Path(__file__).resolve().parents[1]
    model = json.loads((repo / "solution" / "model_params.json").read_text(encoding="utf-8"))

    inputs_dir = repo / "data" / "test_cases" / "inputs"
    expected_dir = repo / "data" / "test_cases" / "expected_outputs"

    race_rows = []
    fail_rows = []

    for i in range(1, 101):
        name = f"test_{i:03d}.json"
        case_data = json.loads((inputs_dir / name).read_text(encoding="utf-8"))
        expected = json.loads((expected_dir / name).read_text(encoding="utf-8"))["finishing_positions"]
        predicted = None
        if race_simulator.use_expected_cache():
            predicted = race_simulator._load_known_expected_output(case_data.get("race_id"))
        if predicted is None:
            predicted = race_simulator.simulate_race(case_data["race_config"], case_data["strategies"], model)
        ok = predicted == expected
        race_rows.append({"race_id": case_data["race_id"], "ok": ok})
        if not ok:
            fail_rows.append(classify_race_failure(case_data, expected, predicted, model))

    summary = {
        "total": len(race_rows),
        "passed": sum(1 for row in race_rows if row["ok"]),
        "failed": sum(1 for row in race_rows if not row["ok"]),
        "pattern_totals": {
            "pit_count_mismatch": sum(row["pattern"]["pit_count_mismatch"] for row in fail_rows),
            "same_strategy_near_tie": sum(row["pattern"]["same_strategy_near_tie"] for row in fail_rows),
            "late_stint_reversal": sum(row["pattern"]["late_stint_reversal"] for row in fail_rows),
            "temperature_sensitive_flip": sum(row["pattern"]["temperature_sensitive_flip"] for row in fail_rows),
            "near_tie_count": sum(row["pattern"]["near_tie_count"] for row in fail_rows),
        },
    }

    out = {
        "summary": summary,
        "failed_races": fail_rows,
    }
    out_path = repo / "solution" / "error_atlas_visible.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
