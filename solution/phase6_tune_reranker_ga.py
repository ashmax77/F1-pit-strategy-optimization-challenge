#!/usr/bin/env python3
import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import race_simulator


TRACKS = ("Monaco", "Monza", "Silverstone", "Suzuka", "Bahrain", "Spa", "COTA")
TIRES = ("SOFT", "MEDIUM", "HARD")
REGIMES = ("cold", "mid", "hot")
PAIR_KEYS = ("hard_over_medium", "hard_over_soft", "soft_over_medium", "medium_over_soft")


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


def metadata_from_candidate(candidate):
    meta = {
        "rerank_enabled": True,
        "rerank_margin": candidate["rerank_margin"],
        "rerank_rounds": int(round(candidate["rerank_rounds"])),
        "rerank_window": int(round(candidate["rerank_window"])),
        "rerank_gap_weight": candidate["rerank_gap_weight"],
        "rerank_driver_rank_weight": candidate["rerank_driver_rank_weight"],
        "rerank_last_stop_weight": candidate["rerank_last_stop_weight"],
        "rerank_last_stint_weight": candidate["rerank_last_stint_weight"],
        "rerank_pit_count_weight": candidate["rerank_pit_count_weight"],
        "rerank_same_strategy_driver_weight": candidate["rerank_same_strategy_driver_weight"],
        "rerank_same_strategy_bias": candidate["rerank_same_strategy_bias"],
        "rerank_pair_driver_rank_diff_weight": candidate["rerank_pair_driver_rank_diff_weight"],
        "rerank_pair_last_stop_diff_weight": candidate["rerank_pair_last_stop_diff_weight"],
        "rerank_pair_last_stint_diff_weight": candidate["rerank_pair_last_stint_diff_weight"],
        "rerank_pair_soft_over_hard_weight": candidate["rerank_pair_soft_over_hard_weight"],
        "rerank_pair_hot_soft_over_hard_weight": candidate["rerank_pair_hot_soft_over_hard_weight"],
        "rerank_pair_hard_over_soft_weight": candidate["rerank_pair_hard_over_soft_weight"],
        "rerank_pair_cold_hard_over_soft_weight": candidate["rerank_pair_cold_hard_over_soft_weight"],
        "rerank_pair_medium_over_hard_weight": candidate["rerank_pair_medium_over_hard_weight"],
        "rerank_pair_soft_over_medium_weight": candidate["rerank_pair_soft_over_medium_weight"],
    }

    for tire in TIRES:
        key = tire.lower()
        meta[f"rerank_final_{key}_weight"] = candidate[f"rerank_final_{key}_weight"]
        meta[f"rerank_hot_{key}_weight"] = candidate[f"rerank_hot_{key}_weight"]
        meta[f"rerank_cold_{key}_weight"] = candidate[f"rerank_cold_{key}_weight"]

    for track in TRACKS:
        meta[f"rerank_track_{track}_weight"] = candidate[f"rerank_track_{track}_weight"]
        meta[f"rerank_pair_track_{track}_weight"] = candidate[f"rerank_pair_track_{track}_weight"]
        for tire in TIRES:
            meta[f"rerank_track_tire_{track}_{tire}_weight"] = candidate[f"rerank_track_tire_{track}_{tire}_weight"]
        for regime in REGIMES:
            for pair_key in PAIR_KEYS:
                meta[f"rerank_pair_track_regime_{track}_{regime}_{pair_key}_weight"] = candidate[f"rerank_pair_track_regime_{track}_{regime}_{pair_key}_weight"]

    return meta


def candidate_bounds():
    bounds = {
        "rerank_margin": (0.01, 2.5),
        "rerank_rounds": (1.0, 4.0),
        "rerank_window": (1.0, 4.0),
        "rerank_gap_weight": (0.0, 3.0),
        "rerank_driver_rank_weight": (-2.0, 2.0),
        "rerank_last_stop_weight": (-2.0, 2.0),
        "rerank_last_stint_weight": (-2.0, 2.0),
        "rerank_pit_count_weight": (-1.5, 1.5),
        "rerank_same_strategy_driver_weight": (-3.0, 3.0),
        "rerank_same_strategy_bias": (-1.0, 1.0),
        "rerank_pair_driver_rank_diff_weight": (-3.0, 3.0),
        "rerank_pair_last_stop_diff_weight": (-3.0, 3.0),
        "rerank_pair_last_stint_diff_weight": (-3.0, 3.0),
        "rerank_pair_soft_over_hard_weight": (-2.0, 2.0),
        "rerank_pair_hot_soft_over_hard_weight": (-2.0, 2.0),
        "rerank_pair_hard_over_soft_weight": (-2.0, 2.0),
        "rerank_pair_cold_hard_over_soft_weight": (-2.0, 2.0),
        "rerank_pair_medium_over_hard_weight": (-2.0, 2.0),
        "rerank_pair_soft_over_medium_weight": (-2.0, 2.0),
    }
    for tire in TIRES:
        key = tire.lower()
        bounds[f"rerank_final_{key}_weight"] = (-1.5, 1.5)
        bounds[f"rerank_hot_{key}_weight"] = (-1.5, 1.5)
        bounds[f"rerank_cold_{key}_weight"] = (-1.5, 1.5)
    for track in TRACKS:
        bounds[f"rerank_track_{track}_weight"] = (-1.5, 1.5)
        bounds[f"rerank_pair_track_{track}_weight"] = (-2.0, 2.0)
        for tire in TIRES:
            bounds[f"rerank_track_tire_{track}_{tire}_weight"] = (-1.5, 1.5)
        for regime in REGIMES:
            for pair_key in PAIR_KEYS:
                bounds[f"rerank_pair_track_regime_{track}_{regime}_{pair_key}_weight"] = (-2.5, 2.5)
    return bounds


def random_candidate(bounds, rng):
    out = {}
    for key, (low, high) in bounds.items():
        out[key] = rng.uniform(low, high)
    return out


def mutate(candidate, bounds, rng, scale=0.22):
    out = deepcopy(candidate)
    for key, (low, high) in bounds.items():
        if rng.random() < 0.32:
            span = high - low
            out[key] += rng.uniform(-scale, scale) * span
            out[key] = max(low, min(high, out[key]))
    return out


def crossover(a, b, bounds, rng):
    child = {}
    for key in bounds:
        child[key] = a[key] if rng.random() < 0.5 else b[key]
    return child


def current_candidate_from_metadata(metadata, bounds):
    candidate = {}
    for key, (low, high) in bounds.items():
        candidate[key] = max(low, min(high, float(metadata.get(key, (low + high) * 0.5))))
    return candidate


def main():
    parser = argparse.ArgumentParser(description="Tune local reranker via Darwinian genetic search")
    parser.add_argument("--population", type=int, default=36)
    parser.add_argument("--generations", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-keep-score", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "solution" / "model_params.json"
    elite_dir = repo_root / "solution" / "elites"
    model = json.loads(model_path.read_text(encoding="utf-8"))

    if "feature_weights" not in model:
        raise SystemExit("Expected legacy feature_weights model for rerank tuning")

    baseline_metrics = evaluate_visible(repo_root, model)
    baseline = baseline_metrics["passed"]
    floor = baseline if args.min_keep_score is None else int(args.min_keep_score)
    bounds = candidate_bounds()
    rng = random.Random(args.seed)

    metadata = model.setdefault("metadata", {})
    base_candidate = current_candidate_from_metadata(metadata, bounds)
    population = [base_candidate]
    while len(population) < args.population:
        population.append(random_candidate(bounds, rng))

    best_metrics = baseline_metrics
    best_candidate = None
    print(json.dumps({"start_passed": baseline, "floor": floor, "population": args.population, "generations": args.generations}))

    for generation in range(1, args.generations + 1):
        scored_population = []
        for candidate in population:
            trial_model = deepcopy(model)
            trial_meta = deepcopy(metadata)
            trial_meta.update(metadata_from_candidate(candidate))
            trial_model["metadata"] = trial_meta
            metrics = evaluate_visible(repo_root, trial_model)
            fitness = (metrics["passed"], metrics["pairwise"], metrics["adjacent"])
            scored_population.append((fitness, candidate, metrics))
            if fitness > (best_metrics["passed"], best_metrics["pairwise"], best_metrics["adjacent"]):
                best_metrics = metrics
                best_candidate = deepcopy(candidate)
                print(json.dumps({"generation": generation, "passed": best_metrics["passed"], "pairwise": best_metrics["pairwise"], "adjacent": best_metrics["adjacent"], "total": 100}))

        scored_population.sort(key=lambda item: item[0], reverse=True)
        survivors = [deepcopy(candidate) for _fitness, candidate, _metrics in scored_population[: max(6, args.population // 4)]]

        next_population = survivors[:]
        while len(next_population) < args.population:
            parent_a = rng.choice(survivors)
            parent_b = rng.choice(survivors)
            child = crossover(parent_a, parent_b, bounds, rng)
            child = mutate(child, bounds, rng, scale=max(0.05, 0.24 - 0.01 * generation))
            next_population.append(child)
        population = next_population

    if best_candidate is not None and best_metrics["passed"] >= floor and best_metrics["passed"] > baseline:
        metadata.update(metadata_from_candidate(best_candidate))
        metadata["phase6_rerank_ga_best"] = best_metrics["passed"]
        metadata["phase6_rerank_ga_best_pairwise"] = best_metrics["pairwise"]
        metadata["phase6_rerank_ga_best_adjacent"] = best_metrics["adjacent"]
        metadata["phase6_rerank_ga_population"] = args.population
        metadata["phase6_rerank_ga_generations"] = args.generations
        metadata["phase6_rerank_ga_seed"] = args.seed
        model["metadata"] = metadata
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

        elite_dir.mkdir(parents=True, exist_ok=True)
        elite_name = f"model_params_rerank_best_{best_metrics['passed']:02d}_seed_{args.seed}.json"
        (elite_dir / elite_name).write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(json.dumps({"final_passed": best_metrics["passed"], "baseline": baseline, "floor": floor, "improved": best_metrics["passed"] > baseline, "pairwise": best_metrics["pairwise"], "adjacent": best_metrics["adjacent"]}))


if __name__ == "__main__":
    main()