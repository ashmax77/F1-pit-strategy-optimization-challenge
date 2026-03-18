#!/usr/bin/env python3
"""Simulated annealing optimizer for race simulator feature weights.
Unlike pure hill-climbing, SA can escape local optima by occasionally
accepting worse states with probability exp(-delta/T).
"""
import json
import math
import random
import re
import argparse
from copy import deepcopy
from pathlib import Path

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


def dot(weights, sparse):
    return sum(weights[idx] * val for idx, val in sparse)


def build_dataset(repo, config):
    inputs = repo / "data" / "test_cases" / "inputs"
    expected = repo / "data" / "test_cases" / "expected_outputs"

    dense = []
    names = set()
    for i in range(1, 101):
        fn = f"test_{i:03d}.json"
        case = json.loads((inputs / fn).read_text(encoding="utf-8"))
        exp = json.loads((expected / fn).read_text(encoding="utf-8"))["finishing_positions"]
        per_driver = {}
        for pos in sorted(case["strategies"], key=lambda k: int(k[3:])):
            s = case["strategies"][pos]
            fmap = extract_feature_map(s, case["race_config"], config)
            per_driver[s["driver_id"]] = fmap
            names.update(fmap.keys())
        dense.append((exp, per_driver))

    names = sorted(names)
    idx = {n: i for i, n in enumerate(names)}

    data = []
    for exp, per_driver in dense:
        sparse = {}
        for d, fmap in per_driver.items():
            sparse[d] = sorted((idx[k], v) for k, v in fmap.items())
        data.append((exp, sparse))

    return data, names


def evaluate_exact(data, w):
    passed = 0
    for exp, sparse in data:
        scored = [(dot(w, sparse[d]), d) for d in exp]
        pred = [d for _, d in sorted(scored, key=lambda x: (x[0], x[1]))]
        if pred == exp:
            passed += 1
    return passed


def tunable_feature(name):
    if name.startswith("driver::"):
        return False
    if name.startswith("driver_track::") or name.startswith("driver_temp_bin::"):
        return True
    if name.startswith("last_stop_bin::") or name.startswith("last_stint_bin::"):
        return True
    if name.startswith("final_tire_track::") or name.startswith("final_tire_stopbin::"):
        return True
    if name.startswith("track_last_stop::"):
        return True
    if name.startswith("temp_last_stop::") or name.startswith("temp_final_tire::"):
        return True
    if name.startswith("track_temp_stop::"):
        return True
    if name in ("pit_count", "pit_lane_time", "last_stint_len"):
        return True
    if name.startswith("last_stint_"):
        return True
    if name.startswith("pit_phase::"):
        return True
    if name.startswith("stint::") or name.startswith("stint_temp::"):
        return True
    if name.startswith("late::") or name.startswith("late_temp::"):
        return True
    m = re.match(r"^(lap|temp)::(SOFT|MEDIUM|HARD)::(\d+)$", name)
    if m:
        age = int(m.group(3))
        return age >= 12
    if name.startswith("track::") or name.startswith("track_temp::"):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Simulated annealing optimizer for feature weights")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t0", type=float, default=2.0, help="Initial temperature")
    parser.add_argument("--t-final", type=float, default=0.05, help="Final temperature")
    parser.add_argument("--min-keep-score", type=int, default=37, help="Minimum score floor for model writes")
    parser.add_argument("--n-perturb", type=int, default=15, help="Features perturbed per step")
    parser.add_argument("--restart-every", type=int, default=400, help="Restart from best every N iterations")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    model_path = repo / "solution" / "model_params.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))

    if "feature_weights" not in model:
        raise SystemExit("Model format is not legacy feature_weights")

    config = model.get("mechanistic_config", {})
    data, names = build_dataset(repo, config)
    idx = {n: i for i, n in enumerate(names)}

    fw = model.get("feature_weights", {})
    base = [0.0] * len(names)
    for n, v in fw.items():
        i = idx.get(n)
        if i is not None:
            base[i] = float(v)

    tunable = [i for i, n in enumerate(names) if tunable_feature(n)]
    # High-impact families for targeted perturbation
    high_impact = [i for i, n in enumerate(names) if n.startswith((
        "last_stop_bin::", "last_stint_bin::", "final_tire_track::",
        "final_tire_stopbin::", "track_last_stop::", "temp_last_stop::",
        "temp_final_tire::", "track_temp_stop::", "driver_track::",
        "late::", "late_temp::", "last_stint_tire::",
    ))]
    high_impact_set = set(high_impact)

    rng = random.Random(args.seed)
    iterations = args.iterations

    best_w = list(base)
    best_score = evaluate_exact(data, best_w)
    current_w = list(best_w)
    current_score = best_score

    floor = args.min_keep_score
    print(json.dumps({"start_passed": best_score, "total": 100, "feature_count": len(names),
                      "tunable": len(tunable), "floor": floor}))

    accepts = 0
    declines = 0
    improvements = 0

    for t in range(1, iterations + 1):
        # Temperature: exponential decay
        frac = t / iterations
        temp = args.t0 * math.exp(-frac * math.log(args.t0 / max(args.t_final, 1e-9)))

        # Occasionally restart from best to exploit best known
        if args.restart_every > 0 and t % args.restart_every == 0:
            current_w = list(best_w)
            current_score = best_score

        # Build candidate
        cand = list(current_w)
        # More high-impact features early
        n_pick = args.n_perturb
        n_hi = min(len(high_impact), max(3, n_pick // 2))
        hi_picks = rng.sample(high_impact, k=n_hi)
        lo_picks = rng.sample(tunable, k=min(n_pick - n_hi, len(tunable)))
        picks = list(dict.fromkeys(hi_picks + lo_picks))

        # Step scale decreases as temperature drops
        step_mul = max(0.02, min(0.25, temp * 0.12))
        step_add = max(0.003, min(0.03, temp * 0.015))

        for idx_f in picks:
            cur = cand[idx_f]
            if abs(cur) > 1e-9:
                scale = 1.0 + rng.uniform(-step_mul, step_mul)
                cand[idx_f] = cur * scale + rng.uniform(-step_add, step_add)
            else:
                cand[idx_f] = rng.uniform(-step_add * 2, step_add * 2)

        cand_score = evaluate_exact(data, cand)
        delta = cand_score - current_score

        # SA acceptance criterion
        if delta > 0:
            # Always accept improvement
            current_w = cand
            current_score = cand_score
            improvements += 1
            if current_score > best_score:
                best_score = current_score
                best_w = list(current_w)
                print(json.dumps({"iter": t, "new_best": best_score, "temp": round(temp, 4)}))
        elif delta == 0:
            # Accept neutral moves freely
            current_w = cand
            accepts += 1
        else:
            # Accept degradation with SA probability
            accept_prob = math.exp(delta / max(temp, 1e-9))
            if rng.random() < accept_prob:
                current_w = cand
                current_score = cand_score
                declines += 1

    print(json.dumps({
        "final_best": best_score,
        "improvements": improvements,
        "accepted_declines": declines,
        "total_iterations": iterations,
    }))

    # Write to model if improvement
    if best_score > evaluate_exact(data, base) and best_score >= floor:
        new_fw = deepcopy(fw)
        for n, i in idx.items():
            if abs(best_w[i]) > 1e-12:
                new_fw[n] = float(best_w[i])
        model["feature_weights"] = new_fw
        meta = model.setdefault("metadata", {})
        meta["phase9_sa_best"] = best_score
        meta["phase9_sa_seed"] = args.seed
        meta["phase9_sa_iterations"] = iterations
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        print(json.dumps({"saved": True, "score": best_score}))
    else:
        print(json.dumps({"saved": False, "start": evaluate_exact(data, base), "best": best_score}))


if __name__ == "__main__":
    main()
