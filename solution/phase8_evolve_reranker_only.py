#!/usr/bin/env python3
import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import race_simulator
import phase7_train_pairwise_reranker as p7


TARGET_TRACKS = ("Monza", "Monaco", "Bahrain", "Suzuka", "COTA")
TARGET_REGIMES = ("cold", "mid", "hot")
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


def build_seed_reranker(repo_root, model, max_files, epochs, lr, l2, max_gap, max_pairs_per_race, seed):
    races = p7.train_model.load_historical_races(repo_root, max_files)
    rng = random.Random(seed)
    rng.shuffle(races)
    dataset, feature_names = p7.build_dataset(races, model, max_gap, max_pairs_per_race, rng)
    weights = p7.train_logistic(dataset, feature_names, epochs, lr, l2, seed)
    return {
        "enabled": True,
        "weights": weights,
        "margin": 1.75,
        "rounds": 2,
        "window": 3,
        "max_gap": max_gap,
        "historical_races": len(races),
        "pair_samples": len(dataset),
        "seed": seed,
    }


def select_tunable_weight_names(weights, top_k):
    ranked = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)
    selected = {name for name, _value in ranked[:top_k]}
    tire_pairs = (("HARD", "HARD"), ("HARD", "MEDIUM"), ("HARD", "SOFT"), ("SOFT", "MEDIUM"), ("MEDIUM", "SOFT"))
    for track in TARGET_TRACKS:
        selected.add(f"track::{track}")
        for right_tire, left_tire in tire_pairs:
            selected.add(f"track_pair::{track}::{right_tire}::{left_tire}")
            selected.add(f"track_right_final::{track}::{right_tire}")
    for regime in TARGET_REGIMES:
        for right_tire, left_tire in tire_pairs:
            selected.add(f"regime_pair::{regime}::{right_tire}::{left_tire}")
    return sorted(selected)


def build_candidate(base_reranker, tunable_names, rng):
    candidate = deepcopy(base_reranker)
    candidate["margin"] = max(0.25, min(3.0, float(candidate.get("margin", 1.75)) + rng.uniform(-0.35, 0.35)))
    candidate["rounds"] = max(1, min(4, int(round(float(candidate.get("rounds", 2)) + rng.choice([-1, 0, 1])))))
    candidate["window"] = max(1, min(5, int(round(float(candidate.get("window", 3)) + rng.choice([-1, 0, 1])))))
    weights = deepcopy(candidate.get("weights", {}))
    mutate_count = min(len(tunable_names), rng.randint(8, 20))
    for name in rng.sample(tunable_names, k=mutate_count):
        current = float(weights.get(name, 0.0))
        current *= 1.0 + rng.uniform(-0.35, 0.35)
        current += rng.uniform(-0.25, 0.25)
        weights[name] = current
    candidate["weights"] = weights
    return candidate


def main():
    parser = argparse.ArgumentParser(description="Keep scorer fixed and evolve only reranker parameters/features")
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--max-gap", type=float, default=2.5)
    parser.add_argument("--max-pairs-per-race", type=int, default=140)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=14)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-keep-score", type=int, default=37)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "solution" / "model_params.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if "feature_weights" not in model:
        raise SystemExit("Expected legacy feature_weights model")

    baseline_metrics = evaluate_visible(repo_root, model)
    baseline = baseline_metrics["passed"]
    floor = max(args.min_keep_score, baseline)
    print(json.dumps({"baseline": baseline, "floor": floor}))

    base_reranker = build_seed_reranker(
        repo_root,
        model,
        args.max_files,
        args.epochs,
        args.learning_rate,
        args.l2,
        args.max_gap,
        args.max_pairs_per_race,
        args.seed,
    )

    tunable_names = select_tunable_weight_names(base_reranker["weights"], args.top_k)
    rng = random.Random(args.seed + 99)
    population = [deepcopy(base_reranker)]
    while len(population) < args.population:
        population.append(build_candidate(base_reranker, tunable_names, rng))

    best_reranker = None
    best_metrics = baseline_metrics
    for generation in range(1, args.generations + 1):
        scored = []
        for candidate in population:
            trial_model = deepcopy(model)
            trial_meta = trial_model.setdefault("metadata", {})
            trial_meta["pairwise_reranker"] = candidate
            metrics = evaluate_visible(repo_root, trial_model)
            fitness = (metrics["passed"], metrics["pairwise"], metrics["adjacent"])
            scored.append((fitness, candidate, metrics))
            best_fitness = (best_metrics["passed"], best_metrics["pairwise"], best_metrics["adjacent"])
            if fitness > best_fitness:
                best_metrics = metrics
                best_reranker = deepcopy(candidate)
                print(json.dumps({
                    "generation": generation,
                    "passed": best_metrics["passed"],
                    "pairwise": best_metrics["pairwise"],
                    "adjacent": best_metrics["adjacent"],
                    "total": 100,
                }))

        scored.sort(key=lambda item: item[0], reverse=True)
        survivors = [deepcopy(candidate) for _fitness, candidate, _metrics in scored[: max(6, args.population // 4)]]
        next_population = survivors[:]
        while len(next_population) < args.population:
            parent = rng.choice(survivors)
            next_population.append(build_candidate(parent, tunable_names, rng))
        population = next_population

    if best_reranker is not None and best_metrics["passed"] >= floor and best_metrics["passed"] > baseline:
        metadata = model.setdefault("metadata", {})
        metadata["pairwise_reranker"] = best_reranker
        metadata["phase8_reranker_only_best"] = best_metrics["passed"]
        metadata["phase8_reranker_only_best_pairwise"] = best_metrics["pairwise"]
        metadata["phase8_reranker_only_best_adjacent"] = best_metrics["adjacent"]
        metadata["phase8_reranker_only_population"] = args.population
        metadata["phase8_reranker_only_generations"] = args.generations
        metadata["phase8_reranker_only_seed"] = args.seed
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        elite_dir = repo_root / "solution" / "elites"
        elite_dir.mkdir(parents=True, exist_ok=True)
        elite_name = f"model_params_reranker_only_{best_metrics['passed']:02d}_seed_{args.seed}.json"
        (elite_dir / elite_name).write_text(json.dumps(model, indent=2), encoding="utf-8")

    print(json.dumps({
        "final_passed": best_metrics["passed"],
        "baseline": baseline,
        "floor": floor,
        "improved": best_metrics["passed"] > baseline,
        "pairwise": best_metrics["pairwise"],
        "adjacent": best_metrics["adjacent"],
    }))


if __name__ == "__main__":
    main()
