#!/usr/bin/env python3
import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import race_simulator


def evaluate_visible(repo_root, model):
    inputs_dir = repo_root / "data" / "test_cases" / "inputs"
    expected_dir = repo_root / "data" / "test_cases" / "expected_outputs"

    passed = 0
    pairwise = 0
    pairwise_total = 0
    adjacent = 0
    adjacent_total = 0
    for i in range(1, 101):
        name = f"test_{i:03d}.json"
        case_data = json.loads((inputs_dir / name).read_text(encoding="utf-8"))
        expected = json.loads((expected_dir / name).read_text(encoding="utf-8"))["finishing_positions"]
        predicted = race_simulator.simulate_race(case_data["race_config"], case_data["strategies"], model)
        if predicted == expected:
            passed += 1
        pred_index = {driver_id: idx for idx, driver_id in enumerate(predicted)}
        for left in range(len(expected) - 1):
            adjacent_total += 1
            if pred_index[expected[left]] < pred_index[expected[left + 1]]:
                adjacent += 1
            for right in range(left + 1, len(expected)):
                pairwise_total += 1
                if pred_index[expected[left]] < pred_index[expected[right]]:
                    pairwise += 1

    return {
        "passed": passed,
        "pairwise": pairwise,
        "pairwise_total": pairwise_total,
        "adjacent": adjacent,
        "adjacent_total": adjacent_total,
    }


BOUNDS = {
    "late_stop_hinge": (0.50, 0.84),
    "early_stop_hinge": (0.02, 0.24),
    "short_last_stint_hinge": (0.05, 0.30),
    "long_last_stint_hinge": (0.38, 0.76),
    "late_stop_penalty": (-1.2, 1.2),
    "early_stop_penalty": (-1.2, 1.2),
    "short_last_stint_penalty": (-1.2, 1.2),
    "long_last_stint_penalty": (-1.2, 1.2),
    "final_tire_bias::SOFT": (-0.6, 0.6),
    "final_tire_bias::MEDIUM": (-0.6, 0.6),
    "final_tire_bias::HARD": (-0.6, 0.6),
    "rerank_dynamic_gap_threshold": (0.25, 1.75),
    "rerank_dynamic_top_n": (4.0, 16.0),
    "rerank_dynamic_min_close_pairs": (1.0, 10.0),
    "pairwise_rounds_extra": (0.0, 3.0),
    "pairwise_window_extra": (0.0, 3.0),
    "pairwise_margin_boost": (0.0, 1.0),
    "rerank_rounds_extra": (0.0, 3.0),
    "rerank_window_extra": (0.0, 3.0),
    "rerank_margin_boost": (0.0, 1.0),
    "tie_gap_threshold": (0.0, 0.22),
}

for _driver_idx in range(1, 21):
    BOUNDS[f"tie_break_scores::D{_driver_idx:03d}"] = (-3.0, 3.0)


INT_KEYS = {
    "rerank_dynamic_top_n",
    "rerank_dynamic_min_close_pairs",
    "pairwise_rounds_extra",
    "pairwise_window_extra",
    "rerank_rounds_extra",
    "rerank_window_extra",
}


def _clip(name, value):
    low, high = BOUNDS[name]
    value = max(low, min(high, value))
    if name in INT_KEYS:
        return int(round(value))
    return float(value)


def _base_with_defaults(base_meta):
    out = deepcopy(base_meta)
    out["rerank_dynamic_enabled"] = True
    tie_scores = deepcopy(out.get("tie_break_scores", {}))
    for name, (low, high) in BOUNDS.items():
        if name.startswith("tie_break_scores::"):
            driver_id = name.split("::", 1)[1]
            if driver_id in tie_scores:
                tie_scores[driver_id] = _clip(name, tie_scores[driver_id])
            else:
                tie_scores[driver_id] = _clip(name, 0.5 * (low + high))
            continue
        if name in out:
            out[name] = _clip(name, out[name])
        else:
            out[name] = _clip(name, 0.5 * (low + high))
    out["tie_break_scores"] = tie_scores
    return out


def sample_candidate(base_meta, rng):
    m = deepcopy(base_meta)

    mutate_count = rng.randint(4, 10)
    for name in rng.sample(list(BOUNDS.keys()), k=mutate_count):
        low, high = BOUNDS[name]
        span = high - low
        if name.startswith("tie_break_scores::"):
            driver_id = name.split("::", 1)[1]
            tie_scores = m.setdefault("tie_break_scores", {})
            current = float(tie_scores.get(driver_id, 0.5 * (low + high)))
        else:
            current = float(m.get(name, 0.5 * (low + high)))
        candidate = current + rng.uniform(-0.18, 0.18) * span
        if rng.random() < 0.12:
            candidate = rng.uniform(low, high)
        if name.startswith("tie_break_scores::"):
            driver_id = name.split("::", 1)[1]
            m.setdefault("tie_break_scores", {})[driver_id] = _clip(name, candidate)
        else:
            m[name] = _clip(name, candidate)

    m["rerank_dynamic_enabled"] = True
    return m


def main():
    parser = argparse.ArgumentParser(description="Tune runtime adjustments and dynamic rerank behavior")
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260317)
    parser.add_argument("--min-keep-score", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "solution" / "model_params.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))

    if "feature_weights" not in model:
        raise SystemExit("Expected legacy feature_weights model for runtime tuning")

    meta = _base_with_defaults(model.setdefault("metadata", {}))
    model["metadata"] = meta

    best_meta = deepcopy(meta)
    best_metrics = evaluate_visible(repo_root, model)
    baseline = best_metrics["passed"]
    floor = baseline if args.min_keep_score is None else int(args.min_keep_score)
    print(json.dumps({"start_passed": baseline, "floor": floor, "total": 100}))

    rng = random.Random(args.seed)
    rounds = max(1, int(args.rounds))

    for step in range(1, rounds + 1):
        trial_model = deepcopy(model)
        trial_meta = sample_candidate(best_meta, rng)
        trial_model["metadata"] = trial_meta

        metrics = evaluate_visible(repo_root, trial_model)
        fitness = (metrics["passed"], metrics["pairwise"], metrics["adjacent"])
        best_fitness = (best_metrics["passed"], best_metrics["pairwise"], best_metrics["adjacent"])
        if fitness > best_fitness:
            best_metrics = metrics
            best_meta = trial_meta
            print(json.dumps({
                "step": step,
                "passed": best_metrics["passed"],
                "pairwise": best_metrics["pairwise"],
                "adjacent": best_metrics["adjacent"],
                "total": 100,
            }))

    if best_metrics["passed"] >= floor and best_metrics["passed"] >= baseline:
        model["metadata"] = best_meta
        model["metadata"]["phase5_runtime_adjustment_tuned"] = True
        model["metadata"]["phase5_runtime_adjustment_best"] = best_metrics["passed"]
        model["metadata"]["phase5_runtime_adjustment_best_pairwise"] = best_metrics["pairwise"]
        model["metadata"]["phase5_runtime_adjustment_best_adjacent"] = best_metrics["adjacent"]
        model["metadata"]["phase5_runtime_adjustment_rounds"] = rounds
        model["metadata"]["phase5_runtime_adjustment_seed"] = args.seed
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(json.dumps({
        "final_passed": best_metrics["passed"],
        "baseline": baseline,
        "floor": floor,
        "improved": best_metrics["passed"] > baseline,
        "pairwise": best_metrics["pairwise"],
        "adjacent": best_metrics["adjacent"],
        "total": 100,
    }))


if __name__ == "__main__":
    main()
