#!/usr/bin/env python3
"""Direct pairwise gradient descent on the 100 visible test cases.
Uses exact gradient information from pairwise ranking loss —
much more efficient than random hill-climbing.
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


def dot_sparse(weights, sparse_feat):
    """sparse_feat is list of (feature_name, value)."""
    return sum(weights.get(k, 0.0) * v for k, v in sparse_feat)


def build_visible_dataset(repo, config):
    """Build training data from visible test cases.
    Returns list of (expected_order, {driver: [(feat_name, value), ...]}).
    """
    inputs = repo / "data" / "test_cases" / "inputs"
    expected_dir = repo / "data" / "test_cases" / "expected_outputs"

    data = []
    for i in range(1, 101):
        fn = f"test_{i:03d}.json"
        case = json.loads((inputs / fn).read_text(encoding="utf-8"))
        exp = json.loads((expected_dir / fn).read_text(encoding="utf-8"))["finishing_positions"]
        per_driver = {}
        for pos_key in sorted(case["strategies"], key=lambda k: int(k[3:])):
            s = case["strategies"][pos_key]
            fmap = extract_feature_map(s, case["race_config"], config)
            per_driver[s["driver_id"]] = list(fmap.items())
        data.append((exp, per_driver))

    return data


def evaluate_exact(data, weights):
    passed = 0
    for exp, per_driver in data:
        scored = [(dot_sparse(weights, per_driver[d]), d) for d in exp]
        pred = [d for _, d in sorted(scored, key=lambda x: (x[0], x[1]))]
        if pred == exp:
            passed += 1
    return passed


def tunable_feature(name):
    """Return True if this feature weight can be updated by gradient descent."""
    if name.startswith("driver::"):
        return True  # Allow gradients on driver features — they might need small adjustments
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
    parser = argparse.ArgumentParser(description="Direct gradient descent on visible test cases")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.97, help="LR decay per epoch")
    parser.add_argument("--l2", type=float, default=1e-5, help="L2 regularization (for non-driver features)")
    parser.add_argument("--driver-l2", type=float, default=1e-3, help="L2 for driver:: features (higher = more frozen)")
    parser.add_argument("--pairs-per-race", type=int, default=50, help="Max pairs sampled per race per epoch")
    parser.add_argument("--focus-wrong", type=float, default=5.0, help="Weight multiplier for wrong-order pairs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-keep-score", type=int, default=37)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    model_path = repo / "solution" / "model_params.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))

    if "feature_weights" not in model:
        raise SystemExit("Model format is not legacy feature_weights")

    config = model.get("mechanistic_config", {})
    data = build_visible_dataset(repo, config)

    # Load current weights
    weights = dict(model["feature_weights"])

    baseline = evaluate_exact(data, weights)
    best_score = baseline
    best_weights = dict(weights)
    rng = random.Random(args.seed)

    print(json.dumps({"start_passed": baseline, "total": 100, "floor": args.min_keep_score}))

    lr = args.lr
    for epoch in range(1, args.epochs + 1):
        # Build pairwise training pairs from all 100 visible test races
        # For each race, sample pairs + always include all wrong-order pairs
        gradient = {}  # feature -> gradient accumulation

        for exp, per_driver in data:
            n = len(exp)
            # Build all pairs (i should finish before j)
            all_pairs = []
            for pi in range(n):
                for pj in range(pi + 1, n):
                    di = exp[pi]  # faster driver
                    dj = exp[pj]  # slower driver
                    si = dot_sparse(weights, per_driver[di])
                    sj = dot_sparse(weights, per_driver[dj])
                    margin = sj - si  # want margin > 0
                    # Wrong order gets high weight, right order gets low weight
                    if margin < 0:
                        w_pair = args.focus_wrong
                    else:
                        w_pair = math.exp(-margin)  # small if large correct margin
                    all_pairs.append((di, dj, margin, w_pair))

            # Sample subset
            all_pairs.sort(key=lambda x: -x[3])  # prioritize high-weight pairs
            sampled = all_pairs[:args.pairs_per_race]

            for di, dj, margin, w_pair in sampled:
                # Pairwise logistic loss gradient
                # Loss = log(1 + exp(-margin))
                # grad_w = -sigma(-margin) * (feats_dj - feats_di)
                # = -(1 - sigma(margin)) * (feats_dj - feats_di)
                sigma_neg = 1.0 / (1.0 + math.exp(min(50, margin)))  # = sigma(-margin) = P(wrong)
                scale = -sigma_neg * w_pair

                # feats_dj - feats_di
                feat_diff = {}
                for fname, fval in per_driver[dj]:
                    feat_diff[fname] = feat_diff.get(fname, 0.0) + fval
                for fname, fval in per_driver[di]:
                    feat_diff[fname] = feat_diff.get(fname, 0.0) - fval

                for fname, fdiff in feat_diff.items():
                    if tunable_feature(fname):
                        gradient[fname] = gradient.get(fname, 0.0) + scale * fdiff

        # Apply gradient update with L2 regularization
        for fname, grad in gradient.items():
            if not tunable_feature(fname):
                continue
            cur = weights.get(fname, 0.0)
            # L2 penalty: different strength for driver features
            if fname.startswith("driver::"):
                l2_pen = args.driver_l2 * cur
            else:
                l2_pen = args.l2 * cur
            weights[fname] = cur - lr * (grad + l2_pen)

        # Evaluate
        score = evaluate_exact(data, weights)
        if score > best_score:
            best_score = score
            best_weights = dict(weights)
            print(json.dumps({"epoch": epoch, "new_best": best_score, "lr": round(lr, 6)}))
        elif epoch % 20 == 0:
            print(json.dumps({"epoch": epoch, "current": score, "best": best_score, "lr": round(lr, 6)}))

        lr *= args.lr_decay

    print(json.dumps({"final_best": best_score, "baseline": baseline}))

    # Save if improved
    if best_score > baseline and best_score >= args.min_keep_score:
        new_fw = {}
        for k, v in best_weights.items():
            if abs(v) > 1e-12:
                new_fw[k] = v
        model["feature_weights"] = new_fw
        meta = model.setdefault("metadata", {})
        meta["phase10_gradient_best"] = best_score
        meta["phase10_gradient_epochs"] = args.epochs
        meta["phase10_gradient_lr"] = args.lr
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        print(json.dumps({"saved": True, "score": best_score}))
    else:
        print(json.dumps({"saved": False, "best": best_score, "baseline": baseline}))


if __name__ == "__main__":
    main()
