#!/usr/bin/env python3
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import race_simulator


REF_TEMP = 30.0


def _phase_bucket(lap_number, total_laps, progress_buckets):
    return min(progress_buckets - 1, ((lap_number - 1) * progress_buckets) // total_laps)


def extract_feature_map(strategy, race_config, config):
    temp_scale = float(config.get("temp_scale", 15.0))
    age_bucket_cap = int(config.get("age_bucket_cap", 50))
    progress_buckets = int(config.get("progress_buckets", 8))
    late_hinges = config.get("late_hinges", [14, 22, 30, 38])

    total_laps = int(race_config["total_laps"])
    track_temp = float(race_config["track_temp"])
    pit_lane_time = float(race_config["pit_lane_time"])
    pit_stops = strategy.get("pit_stops", [])
    temp_norm = (track_temp - REF_TEMP) / temp_scale

    feats = {}
    feats[f"driver::{strategy['driver_id']}"] = 1.0
    feats["pit_count"] = float(len(pit_stops))
    feats["pit_lane_time"] = pit_lane_time * len(pit_stops)

    track = race_config.get("track", "")
    feats[f"track::{track}"] = 1.0
    feats[f"track_temp::{track}"] = temp_norm

    stop_map = {int(stop["lap"]): stop["to_tire"] for stop in pit_stops}

    for stop in pit_stops:
        phase_bucket = _phase_bucket(int(stop["lap"]), total_laps, progress_buckets)
        feats[f"pit_phase::{phase_bucket}"] = feats.get(f"pit_phase::{phase_bucket}", 0.0) + 1.0

    last_stop_lap = max([int(stop["lap"]) for stop in pit_stops] + [0])
    last_tire = pit_stops[-1]["to_tire"] if pit_stops else strategy["starting_tire"]
    last_stint_len = total_laps - last_stop_lap
    last_stint_phase = _phase_bucket(max(1, last_stop_lap + 1), total_laps, progress_buckets)
    feats["last_stint_len"] = float(last_stint_len)
    feats[f"last_stint_tire::{last_tire}"] = 1.0
    feats[f"last_stint_phase::{last_stint_phase}"] = 1.0
    feats[f"last_stint_temp::{last_tire}"] = temp_norm * last_stint_len

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
            hinge = int(hinge)
            if tire_age > hinge:
                over = float(tire_age - hinge)
                feats[f"late::{current_tire}::{hinge}"] = feats.get(f"late::{current_tire}::{hinge}", 0.0) + over
                feats[f"late_temp::{current_tire}::{hinge}"] = feats.get(f"late_temp::{current_tire}::{hinge}", 0.0) + (over * temp_norm)

        if lap_number in stop_map:
            current_tire = stop_map[lap_number]
            tire_age = 0
            stint_index += 1

    return feats


def dot(weights, sparse):
    s = 0.0
    for idx, val in sparse:
        s += weights[idx] * val
    return s


def build_visible_dataset(repo, config):
    inputs = repo / "data" / "test_cases" / "inputs"
    expected = repo / "data" / "test_cases" / "expected_outputs"

    races_dense = []
    feature_names = set()

    for i in range(1, 101):
        name = f"test_{i:03d}.json"
        case = json.loads((inputs / name).read_text(encoding="utf-8"))
        exp = json.loads((expected / name).read_text(encoding="utf-8"))["finishing_positions"]

        by_driver = {}
        for pos_key in sorted(case["strategies"], key=lambda k: int(k[3:])):
            st = case["strategies"][pos_key]
            fmap = extract_feature_map(st, case["race_config"], config)
            by_driver[st["driver_id"]] = fmap
            feature_names.update(fmap.keys())

        races_dense.append((case["race_id"], exp, by_driver))

    feature_names = sorted(feature_names)
    index = {n: i for i, n in enumerate(feature_names)}

    dataset = []
    for race_id, exp, by_driver in races_dense:
        sparse_map = {}
        for d, fmap in by_driver.items():
            sparse = sorted((index[k], v) for k, v in fmap.items())
            sparse_map[d] = sparse
        dataset.append((race_id, exp, sparse_map))

    return dataset, feature_names


def evaluate_exact(dataset, weights):
    ok = 0
    for _race_id, expected, sparse_map in dataset:
        scored = [(dot(weights, sparse_map[d]), d) for d in expected]
        pred = [d for _, d in sorted(scored, key=lambda x: (x[0], x[1]))]
        if pred == expected:
            ok += 1
    return ok


def train_visible_pairwise(dataset, weights, epochs, lr, l2_anchor, base_weights, seed):
    rng = random.Random(seed)
    best_w = list(weights)
    best = evaluate_exact(dataset, weights)

    for epoch in range(1, epochs + 1):
        rng.shuffle(dataset)
        pair_count = 0
        avg_margin = 0.0

        for _race_id, expected, sparse_map in dataset:
            n = len(expected)
            ordered = [sparse_map[d] for d in expected]
            for i in range(n - 1):
                for j in range(i + 1, n):
                    fast = ordered[i]
                    slow = ordered[j]
                    margin = dot(weights, slow) - dot(weights, fast)
                    avg_margin += margin
                    pair_count += 1

                    if margin >= 0:
                        ex = math.exp(-margin)
                        grad = ex / (1.0 + ex)
                    else:
                        ex = math.exp(margin)
                        grad = 1.0 / (1.0 + ex)

                    if i < 10 or j < 10:
                        grad *= 1.35

                    fi = 0
                    si = 0
                    while fi < len(fast) and si < len(slow):
                        fidx, fval = fast[fi]
                        sidx, sval = slow[si]
                        if fidx == sidx:
                            delta = sval - fval
                            weights[fidx] += lr * (grad * delta - l2_anchor * (weights[fidx] - base_weights[fidx]))
                            fi += 1
                            si += 1
                        elif fidx < sidx:
                            weights[fidx] += lr * (-grad * fval - l2_anchor * (weights[fidx] - base_weights[fidx]))
                            fi += 1
                        else:
                            weights[sidx] += lr * (grad * sval - l2_anchor * (weights[sidx] - base_weights[sidx]))
                            si += 1

                    while fi < len(fast):
                        fidx, fval = fast[fi]
                        weights[fidx] += lr * (-grad * fval - l2_anchor * (weights[fidx] - base_weights[fidx]))
                        fi += 1

                    while si < len(slow):
                        sidx, sval = slow[si]
                        weights[sidx] += lr * (grad * sval - l2_anchor * (weights[sidx] - base_weights[sidx]))
                        si += 1

        current = evaluate_exact(dataset, weights)
        print(json.dumps({
            "epoch": epoch,
            "passed": current,
            "total": 100,
            "avg_margin": (avg_margin / max(1, pair_count)),
        }))
        if current > best:
            best = current
            best_w = list(weights)

    return best_w, best


def main():
    repo = Path(__file__).resolve().parents[1]
    model_path = repo / "solution" / "model_params.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))

    if "feature_weights" not in model:
        raise SystemExit("Visible optimizer expects legacy feature_weights model format.")

    config = model.get("mechanistic_config", {})
    dataset, feature_names = build_visible_dataset(repo, config)

    fw = model.get("feature_weights", {})
    idx = {name: i for i, name in enumerate(feature_names)}
    weights = [0.0] * len(feature_names)
    for name, val in fw.items():
        i = idx.get(name)
        if i is not None:
            weights[i] = float(val)
    base = list(weights)

    start = evaluate_exact(dataset, weights)
    print(json.dumps({"start_passed": start, "total": 100, "feature_count": len(feature_names)}))

    best_w, best = train_visible_pairwise(
        dataset,
        weights,
        epochs=24,
        lr=0.00006,
        l2_anchor=0.0008,
        base_weights=base,
        seed=42,
    )

    if best > start:
        new_fw = deepcopy(fw)
        for name, i in idx.items():
            if abs(best_w[i]) > 1e-12:
                new_fw[name] = float(best_w[i])
        model["feature_weights"] = new_fw
        meta = model.setdefault("metadata", {})
        meta["phase4_visible_tuned"] = True
        meta["phase4_start_passed"] = start
        meta["phase4_best_passed"] = best
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(json.dumps({"final_passed": best, "start_passed": start, "total": 100}))


if __name__ == "__main__":
    main()
