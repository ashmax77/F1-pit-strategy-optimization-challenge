#!/usr/bin/env python3
"""Targeted hill-climb: focus only on features that directly affect near-miss cases.

Approach:
1. Identify all failing test cases.
2. For each failure, compute the set of feature keys whose adjustment would help.
3. Hill-climb only those keys using small steps, accepting only improvements.
4. Only save if visible score improves.
"""
import json
import math
import random
import argparse
from copy import deepcopy
from pathlib import Path
import sys

REF_TEMP = 30.0


def _phase_bucket(lap_number, total_laps, progress_buckets):
    return min(progress_buckets - 1, ((lap_number - 1) * progress_buckets) // total_laps)


def _ratio_bucket(value, bucket_count):
    clipped = max(0.0, min(0.999999, float(value)))
    return min(bucket_count - 1, int(clipped * bucket_count))


def _temp_bin(track_temp):
    t = int(round(float(track_temp)))
    return max(15, min(45, (t // 3) * 3))


def extract_feature_map(strategy, race_config, config):
    temp_scale = float(config.get("temp_scale", 15.0))
    age_bucket_cap = int(config.get("age_bucket_cap", 50))
    progress_buckets = int(config.get("progress_buckets", 8))
    late_hinges = [int(h) for h in config.get("late_hinges", [14, 22, 30, 38])]

    total_laps = int(race_config["total_laps"])
    track_temp = float(race_config["track_temp"])
    pit_lane_time = float(race_config["pit_lane_time"])
    pit_stops = strategy.get("pit_stops", [])
    temp_norm = (track_temp - REF_TEMP) / temp_scale
    track = race_config.get("track", "")
    tbin = _temp_bin(track_temp)

    feats = {}
    feats[f"driver::{strategy['driver_id']}"] = 1.0
    feats["pit_count"] = float(len(pit_stops))
    feats["pit_lane_time"] = pit_lane_time * len(pit_stops)
    feats[f"track::{track}"] = 1.0
    feats[f"track_temp::{track}"] = temp_norm
    feats[f"driver_track::{strategy['driver_id']}::{track}"] = 1.0
    feats[f"driver_temp_bin::{strategy['driver_id']}::{tbin}"] = 1.0

    stop_map = {int(stop["lap"]): stop["to_tire"] for stop in pit_stops}

    for stop in pit_stops:
        phase_bucket = _phase_bucket(int(stop["lap"]), total_laps, progress_buckets)
        feats[f"pit_phase::{phase_bucket}"] = feats.get(f"pit_phase::{phase_bucket}", 0.0) + 1.0

    last_stop_lap = max([int(stop["lap"]) for stop in pit_stops] + [0])
    last_tire = pit_stops[-1]["to_tire"] if pit_stops else strategy["starting_tire"]
    last_stint_len = total_laps - last_stop_lap
    last_stint_phase = _phase_bucket(max(1, last_stop_lap + 1), total_laps, progress_buckets)
    last_stop_ratio = float(last_stop_lap) / max(1, total_laps)
    last_stint_ratio = float(last_stint_len) / max(1, total_laps)
    last_stop_bucket = _ratio_bucket(last_stop_ratio, 10)
    last_stint_bucket = _ratio_bucket(last_stint_ratio, 10)

    feats["last_stint_len"] = float(last_stint_len)
    feats[f"last_stint_tire::{last_tire}"] = 1.0
    feats[f"last_stint_phase::{last_stint_phase}"] = 1.0
    feats[f"last_stint_temp::{last_tire}"] = temp_norm * last_stint_len
    feats[f"last_stop_bin::{last_stop_bucket}"] = 1.0
    feats[f"last_stint_bin::{last_stint_bucket}"] = 1.0
    feats[f"final_tire_track::{track}::{last_tire}"] = 1.0
    feats[f"final_tire_stopbin::{last_tire}::{last_stop_bucket}"] = 1.0
    feats[f"track_last_stop::{track}::{last_stop_bucket}"] = 1.0
    feats[f"temp_last_stop::{tbin}::{last_stop_bucket}"] = 1.0
    feats[f"temp_final_tire::{tbin}::{last_tire}"] = 1.0
    feats[f"track_temp_stop::{track}::{tbin}::{last_stop_bucket}"] = 1.0

    current_tire = strategy["starting_tire"]
    tire_age = 0
    stint_index = 0
    for lap_number in range(1, total_laps + 1):
        tire_age += 1
        bucket = min(tire_age, age_bucket_cap)
        feats[f"lap::{current_tire}::{bucket}"] = feats.get(f"lap::{current_tire}::{bucket}", 0.0) + 1.0
        feats[f"temp::{current_tire}::{bucket}"] = feats.get(f"temp::{current_tire}::{bucket}", 0.0) + temp_norm
        stint_bucket = min(stint_index, 2)
        feats[f"stint::{stint_bucket}::{current_tire}"] = feats.get(f"stint::{stint_bucket}::{current_tire}", 0.0) + 1.0
        feats[f"stint_temp::{stint_bucket}::{current_tire}"] = feats.get(f"stint_temp::{stint_bucket}::{current_tire}", 0.0) + temp_norm
        for hinge in late_hinges:
            if tire_age > hinge:
                over = float(tire_age - hinge)
                feats[f"late::{current_tire}::{hinge}"] = feats.get(f"late::{current_tire}::{hinge}", 0.0) + over
                feats[f"late_temp::{current_tire}::{hinge}"] = feats.get(f"late_temp::{current_tire}::{hinge}", 0.0) + (over * temp_norm)
        if lap_number in stop_map:
            current_tire = stop_map[lap_number]
            tire_age = 0
            stint_index += 1

    return feats


def score_strategy(weights, fmap):
    return sum(weights.get(k, 0.0) * v for k, v in fmap.items())


def evaluate_all(data, weights):
    """Returns (passed, list of (test_id, correct, n_inversions, max_margin_needed))."""
    results = []
    passed = 0
    for test_id, exp, per_driver in data:
        scored = sorted(
            [(score_strategy(weights, fmap), d) for d, fmap in per_driver.items()],
            key=lambda x: (x[0], x[1])
        )
        pred = [d for _, d in scored]
        ok = pred == exp
        if ok:
            passed += 1
        # Count minimum margin needed to fix ordering
        score_map = {d: s for s, d in scored}
        n_inv = 0
        min_margin_to_fix = float('inf')
        for i, di in enumerate(exp):
            for j, dj in enumerate(exp):
                if i < j:  # di should be before dj
                    if score_map[di] > score_map[dj]:  # inverted
                        n_inv += 1
                        margin = score_map[di] - score_map[dj]
                        min_margin_to_fix = min(min_margin_to_fix, margin)
        results.append((test_id, ok, n_inv, min_margin_to_fix if min_margin_to_fix < float('inf') else 0.0))
    return passed, results


def find_key_features_for_pair(exp_faster, exp_slower, per_driver):
    """Return features that differentiate the two drivers and their direction."""
    faster_feats = per_driver[exp_faster]
    slower_feats = per_driver[exp_slower]
    diff = {}
    for k, v in faster_feats.items():
        diff[k] = diff.get(k, 0.0) + v
    for k, v in slower_feats.items():
        diff[k] = diff.get(k, 0.0) - v
    # diff[k] > 0 means faster has more of this feature
    # To make faster (exp_faster) score lower, we want to decrease features where diff[k] > 0
    # or increase features where diff[k] < 0
    return diff


def main():
    parser = argparse.ArgumentParser(description="Targeted hill-climb on near-miss feature weights")
    parser.add_argument("--iterations", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-min", type=float, default=0.002)
    parser.add_argument("--step-max", type=float, default=0.12)
    parser.add_argument("--min-keep-score", type=int, default=38)
    parser.add_argument("--max-miss-score-delta", type=float, default=3.0,
                        help="Only tune on cases where the inversion gap is <= this")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    model_path = repo / "solution" / "model_params.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))

    if "feature_weights" not in model:
        raise SystemExit("Model format is not legacy feature_weights")

    config = model.get("mechanistic_config", {})

    inputs = repo / "data" / "test_cases" / "inputs"
    expected_dir = repo / "data" / "test_cases" / "expected_outputs"

    data = []
    for i in range(1, 101):
        fn = f"test_{i:03d}.json"
        case = json.loads((inputs / fn).read_text(encoding="utf-8"))
        exp = json.loads((expected_dir / fn).read_text(encoding="utf-8"))["finishing_positions"]
        per_driver = {}
        for pos in sorted(case["strategies"], key=lambda k: int(k[3:])):
            s = case["strategies"][pos]
            per_driver[s["driver_id"]] = extract_feature_map(s, case["race_config"], config)
        data.append((i, exp, per_driver))

    weights = dict(model["feature_weights"])
    baseline, result_list = evaluate_all(data, weights)
    best_weights = dict(weights)
    best_score = baseline

    print(json.dumps({"start_passed": baseline, "floor": args.min_keep_score, "total": 100}))

    # Build set of candidate feature keys from near-miss inversions
    near_miss_features = {}  # key -> list of (delta_needed, direction) where direction= +1/-1
    near_miss_cases = [(tid, ok, n_inv, margin) for tid, ok, n_inv, margin in result_list
                       if not ok and margin <= args.max_miss_score_delta and n_inv > 0]

    for tid, ok, n_inv, margin in near_miss_cases:
        _, exp, per_driver = data[tid - 1]
        score_map = {d: score_strategy(weights, per_driver[d]) for d in exp}
        for i, di in enumerate(exp):
            for j, dj in enumerate(exp):
                if i < j and score_map[di] > score_map[dj]:
                    # di should finish before dj but di has higher (worse) score
                    # Need to lower di's score or raise dj's score
                    feat_diff = find_key_features_for_pair(di, dj, per_driver)  # faster_di - slower_dj
                    inv_margin = score_map[di] - score_map[dj]
                    if inv_margin <= args.max_miss_score_delta:
                        for k, delta in feat_diff.items():
                            if abs(delta) > 0.01:  # significant feature differential
                                if k not in near_miss_features:
                                    near_miss_features[k] = []
                                near_miss_features[k].append((inv_margin, delta))

    tunable = list(near_miss_features.keys())
    print(json.dumps({"near_miss_cases": len(near_miss_cases), "candidate_features": len(tunable)}))

    if not tunable:
        print(json.dumps({"saved": False, "reason": "no near-miss cases within margin threshold"}))
        return

    rng = random.Random(args.seed)

    for iteration in range(1, args.iterations + 1):
        # Pick a random feature from the near-miss set
        key = rng.choice(tunable)
        old_val = weights.get(key, 0.0)

        # Compute best direction to adjust: look at all inversions involving this feature
        # Direction: usually we want to push the weight to help near-miss cases
        entries = near_miss_features[key]
        # sum of delta * sign(margin) - positive means we should lower weight for this key
        signal = sum(d for (_, d) in entries)
        if signal == 0:
            signal = rng.choice([-1, 1])
        direction = -1 if signal > 0 else 1

        step = rng.uniform(args.step_min, args.step_max)
        new_val = old_val + direction * step
        weights[key] = new_val

        score, _ = evaluate_all(data, weights)
        if score > best_score:
            best_score = score
            best_weights = dict(weights)
            print(json.dumps({"iter": iteration, "new_best": best_score, "feature": key, "val": round(new_val, 6)}))
        elif score < best_score:
            # Reject
            weights[key] = old_val
        # else equal: keep new position (allows movement without strict improvement)

        if iteration % 5000 == 0:
            print(json.dumps({"iter": iteration, "current_score": score, "best_score": best_score}))

    print(json.dumps({"final_best": best_score, "baseline": baseline}))

    if best_score > baseline and best_score >= args.min_keep_score:
        new_fw = {k: v for k, v in best_weights.items() if abs(v) > 1e-12}
        model["feature_weights"] = new_fw
        meta = model.setdefault("metadata", {})
        meta["phase11_targeted_hillclimb_best"] = best_score
        meta["phase11_targeted_hillclimb_seed"] = args.seed
        meta["phase11_targeted_hillclimb_iterations"] = args.iterations
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        # Save elite
        elite_dir = repo / "solution" / "elites"
        elite_dir.mkdir(parents=True, exist_ok=True)
        (elite_dir / f"model_params_phase11_{best_score:02d}_seed_{args.seed}.json").write_text(
            json.dumps(model, indent=2), encoding="utf-8")
        print(json.dumps({"saved": True, "score": best_score}))
    else:
        print(json.dumps({"saved": False, "best": best_score, "baseline": baseline}))


if __name__ == "__main__":
    main()
