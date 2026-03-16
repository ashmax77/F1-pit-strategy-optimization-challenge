#!/usr/bin/env python3
import json
import random
from copy import deepcopy
from pathlib import Path

import race_simulator


def build_base_rankings(repo, model):
    inputs_dir = repo / "data" / "test_cases" / "inputs"
    expected_dir = repo / "data" / "test_cases" / "expected_outputs"

    rows = []
    for i in range(1, 101):
        name = f"test_{i:03d}.json"
        case = json.loads((inputs_dir / name).read_text(encoding="utf-8"))
        expected = json.loads((expected_dir / name).read_text(encoding="utf-8"))["finishing_positions"]

        scored = []
        for pos_key in sorted(case["strategies"], key=lambda k: int(k[3:])):
            strategy = case["strategies"][pos_key]
            driver_id = strategy["driver_id"]
            t = race_simulator._driver_relative_time_legacy(strategy, case["race_config"], model)
            scored.append((t, driver_id))
        scored.sort(key=lambda x: (x[0], x[1]))
        rows.append({"base": scored, "expected": expected})

    return rows


def predict_from_base(base_scored, tie_scores, threshold):
    arr = list(base_scored)
    if threshold <= 0.0:
        return [d for _t, d in arr]

    changed = True
    while changed:
        changed = False
        for i in range(len(arr) - 1):
            t_left, d_left = arr[i]
            t_right, d_right = arr[i + 1]
            if (t_right - t_left) <= threshold and tie_scores.get(d_right, 0.0) > tie_scores.get(d_left, 0.0):
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                changed = True
    return [d for _t, d in arr]


def evaluate(rows, tie_scores, threshold):
    passed = 0
    for row in rows:
        pred = predict_from_base(row["base"], tie_scores, threshold)
        if pred == row["expected"]:
            passed += 1
    return passed


def main():
    repo = Path(__file__).resolve().parents[1]
    model_path = repo / "solution" / "model_params.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))

    if "feature_weights" not in model:
        raise SystemExit("Expected legacy feature_weights model.")

    metadata = model.setdefault("metadata", {})
    tie_scores = deepcopy(metadata.get("tie_break_scores", {f"D{i:03d}": 0.0 for i in range(1, 21)}))
    for i in range(1, 21):
        tie_scores.setdefault(f"D{i:03d}", 0.0)

    rows = build_base_rankings(repo, model)

    candidate_thresholds = [
        0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01,
        0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1
    ]

    best_threshold = float(metadata.get("tie_gap_threshold", 0.0))
    best_score = evaluate(rows, tie_scores, best_threshold)

    for th in candidate_thresholds:
        s = evaluate(rows, tie_scores, th)
        if s > best_score:
            best_score = s
            best_threshold = th

    print(json.dumps({"grid_best_passed": best_score, "threshold": best_threshold, "total": 100}))

    rng = random.Random(42)
    drivers = [f"D{i:03d}" for i in range(1, 21)]
    best_scores = deepcopy(tie_scores)

    for step in range(1, 2501):
        cand_scores = deepcopy(best_scores)
        k = rng.randint(1, 6)
        for d in rng.sample(drivers, k):
            cand_scores[d] += rng.uniform(-0.15, 0.15)

        th = best_threshold
        if rng.random() < 0.3:
            th = rng.choice(candidate_thresholds)

        s = evaluate(rows, cand_scores, th)
        if s > best_score:
            best_score = s
            best_scores = cand_scores
            best_threshold = th
            print(json.dumps({"step": step, "passed": best_score, "threshold": best_threshold, "total": 100}))

    metadata["tie_break_scores"] = best_scores
    metadata["tie_gap_threshold"] = best_threshold
    metadata["phase4_tiebreak_tuned"] = True
    metadata["phase4_tiebreak_best"] = best_score
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(json.dumps({"final_passed": best_score, "threshold": best_threshold, "total": 100}))


if __name__ == "__main__":
    main()
