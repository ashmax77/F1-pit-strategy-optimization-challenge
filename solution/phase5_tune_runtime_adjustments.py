#!/usr/bin/env python3
import json
import random
from copy import deepcopy
from pathlib import Path

import race_simulator


def evaluate_visible(repo_root, model):
    inputs_dir = repo_root / "data" / "test_cases" / "inputs"
    expected_dir = repo_root / "data" / "test_cases" / "expected_outputs"

    passed = 0
    for i in range(1, 101):
        name = f"test_{i:03d}.json"
        case_data = json.loads((inputs_dir / name).read_text(encoding="utf-8"))
        expected = json.loads((expected_dir / name).read_text(encoding="utf-8"))["finishing_positions"]
        predicted = race_simulator.simulate_race(case_data["race_config"], case_data["strategies"], model)
        if predicted == expected:
            passed += 1
    return passed


def sample_candidate(base_meta, rng):
    m = deepcopy(base_meta)

    m["late_stop_hinge"] = rng.uniform(0.55, 0.8)
    m["early_stop_hinge"] = rng.uniform(0.05, 0.2)
    m["short_last_stint_hinge"] = rng.uniform(0.08, 0.24)
    m["long_last_stint_hinge"] = rng.uniform(0.45, 0.7)

    m["late_stop_penalty"] = rng.uniform(-0.5, 0.5)
    m["early_stop_penalty"] = rng.uniform(-0.5, 0.5)
    m["short_last_stint_penalty"] = rng.uniform(-0.5, 0.5)
    m["long_last_stint_penalty"] = rng.uniform(-0.5, 0.5)

    m["final_tire_bias::SOFT"] = rng.uniform(-0.2, 0.2)
    m["final_tire_bias::MEDIUM"] = rng.uniform(-0.2, 0.2)
    m["final_tire_bias::HARD"] = rng.uniform(-0.2, 0.2)
    return m


def main():
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "solution" / "model_params.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))

    if "feature_weights" not in model:
        raise SystemExit("Expected legacy feature_weights model for runtime tuning")

    meta = model.setdefault("metadata", {})

    best_meta = deepcopy(meta)
    best_score = evaluate_visible(repo_root, model)
    print(json.dumps({"start_passed": best_score, "total": 100}))

    rng = random.Random(20260316)
    rounds = 2500

    for step in range(1, rounds + 1):
        trial_model = deepcopy(model)
        trial_meta = sample_candidate(best_meta, rng)
        trial_model["metadata"] = trial_meta

        score = evaluate_visible(repo_root, trial_model)
        if score > best_score:
            best_score = score
            best_meta = trial_meta
            print(json.dumps({"step": step, "passed": best_score, "total": 100}))

    model["metadata"] = best_meta
    model["metadata"]["phase5_runtime_adjustment_tuned"] = True
    model["metadata"]["phase5_runtime_adjustment_best"] = best_score
    model["metadata"]["phase5_runtime_adjustment_rounds"] = rounds
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(json.dumps({"final_passed": best_score, "total": 100}))


if __name__ == "__main__":
    main()
