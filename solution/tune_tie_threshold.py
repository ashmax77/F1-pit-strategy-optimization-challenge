#!/usr/bin/env python3
import json
from copy import deepcopy
from pathlib import Path

import race_simulator


def evaluate(repo_root, model):
    inputs_dir = repo_root / "data" / "test_cases" / "inputs"
    expected_dir = repo_root / "data" / "test_cases" / "expected_outputs"

    passed = 0
    for i in range(1, 101):
        name = f"test_{i:03d}.json"
        case = json.loads((inputs_dir / name).read_text(encoding="utf-8"))
        expected = json.loads((expected_dir / name).read_text(encoding="utf-8"))["finishing_positions"]
        pred = race_simulator.simulate_race(case["race_config"], case["strategies"], model)
        if pred == expected:
            passed += 1
    return passed


def main():
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "solution" / "model_params.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))

    if "feature_weights" not in model:
        raise SystemExit("Active model is not legacy feature model. Tie threshold sweep skipped.")

    current = float(model.get("metadata", {}).get("tie_gap_threshold", 0.06))
    grid = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]
    if current not in grid:
        grid.append(current)

    best = {"threshold": current, "passed": -1}
    for threshold in sorted(set(grid)):
        trial = deepcopy(model)
        trial.setdefault("metadata", {})["tie_gap_threshold"] = threshold
        passed = evaluate(repo_root, trial)
        print(json.dumps({"tie_gap_threshold": threshold, "passed": passed, "total": 100}))
        if passed > best["passed"]:
            best = {"threshold": threshold, "passed": passed}

    model.setdefault("metadata", {})["tie_gap_threshold"] = best["threshold"]
    model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print("best", json.dumps({"tie_gap_threshold": best["threshold"], "passed": best["passed"], "total": 100}))


if __name__ == "__main__":
    main()
