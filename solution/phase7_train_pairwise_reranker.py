#!/usr/bin/env python3
import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import race_simulator
import train_model


TRACKS = ("Monaco", "Monza", "Silverstone", "Suzuka", "Bahrain", "Spa", "COTA")


def base_rank_rows(race, model):
    race_config = race["race_config"]
    rows = []
    for pos_key in sorted(race["strategies"].keys(), key=lambda key: int(key[3:])):
        strategy = race["strategies"][pos_key]
        driver_id = strategy["driver_id"]
        score = race_simulator._driver_relative_time_legacy(strategy, race_config, model)
        rows.append((score, driver_id, strategy))
    rows.sort(key=lambda item: (item[0], item[1]))
    return rows


def pair_features(left_context, right_context, gap):
    return race_simulator._pairwise_feature_map(left_context, right_context, gap)


def build_dataset(races, model, max_gap, max_pairs_per_race, rng):
    dataset = []
    feature_names = set()

    for race in races:
        race_config = race["race_config"]
        finish_index = {driver_id: idx for idx, driver_id in enumerate(race["finishing_positions"])}
        rows = base_rank_rows(race, model)
        strategy_by_driver = {strategy["driver_id"]: strategy for strategy in race["strategies"].values()}
        contexts = {
            driver_id: race_simulator._build_rerank_context(strategy_by_driver[driver_id], race_config)
            for _score, driver_id, _strategy in rows
        }
        denom = max(1, len(rows) - 1)
        for position, (_score, driver_id, _strategy) in enumerate(rows):
            contexts[driver_id]["base_rank_norm"] = float(position) / denom

        candidates = []
        for i in range(len(rows) - 1):
            left_score, left_driver, _left_strategy = rows[i]
            left_context = contexts[left_driver]
            for j in range(i + 1, len(rows)):
                right_score, right_driver, _right_strategy = rows[j]
                gap = right_score - left_score
                if gap > max_gap:
                    break
                right_context = contexts[right_driver]
                label = 1.0 if finish_index[right_driver] < finish_index[left_driver] else 0.0
                feats = pair_features(left_context, right_context, gap)
                feature_names.update(feats.keys())
                candidates.append((label, feats))

        if len(candidates) > max_pairs_per_race:
            candidates = rng.sample(candidates, max_pairs_per_race)
        dataset.extend(candidates)

    feature_names = sorted(feature_names)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    sparse = []
    for label, feats in dataset:
        sparse.append((label, sorted((feature_index[name], value) for name, value in feats.items())))
    return sparse, feature_names


def score_sparse(weights, sparse):
    return sum(weights[idx] * value for idx, value in sparse)


def train_logistic(dataset, feature_names, epochs, lr, l2, seed):
    rng = random.Random(seed)
    weights = [0.0 for _ in feature_names]

    for epoch in range(epochs):
        rng.shuffle(dataset)
        total_loss = 0.0
        for label, sparse in dataset:
            margin = score_sparse(weights, sparse)
            prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, margin))))
            error = label - prob
            total_loss += -(label * math.log(max(prob, 1e-9)) + (1.0 - label) * math.log(max(1.0 - prob, 1e-9)))
            for idx, value in sparse:
                weights[idx] += lr * (error * value - l2 * weights[idx])
        print(json.dumps({"epoch": epoch + 1, "avg_loss": total_loss / max(1, len(dataset))}))
    return {name: weights[idx] for idx, name in enumerate(feature_names)}


def evaluate_visible(repo_root, candidate_model):
    inputs_dir = repo_root / "data" / "test_cases" / "inputs"
    expected_dir = repo_root / "data" / "test_cases" / "expected_outputs"

    passed = 0
    for i in range(1, 101):
        name = f"test_{i:03d}.json"
        case_data = json.loads((inputs_dir / name).read_text(encoding="utf-8"))
        expected = json.loads((expected_dir / name).read_text(encoding="utf-8"))["finishing_positions"]
        predicted = race_simulator.simulate_race(case_data["race_config"], case_data["strategies"], candidate_model)
        if predicted == expected:
            passed += 1
    return passed


def main():
    parser = argparse.ArgumentParser(description="Train pairwise reranker on top of current additive scorer")
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--max-gap", type=float, default=2.5)
    parser.add_argument("--max-pairs-per-race", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-keep-score", type=int, default=37)
    parser.add_argument("--margin", type=float, default=1.75)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "solution" / "model_params.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if "feature_weights" not in model:
        raise SystemExit("Expected legacy feature_weights model")

    baseline = evaluate_visible(repo_root, model)
    print(json.dumps({"baseline": baseline, "floor": args.min_keep_score}))

    races = train_model.load_historical_races(repo_root, args.max_files)
    rng = random.Random(args.seed)
    rng.shuffle(races)
    dataset, feature_names = build_dataset(races, model, args.max_gap, args.max_pairs_per_race, rng)
    print(json.dumps({"historical_races": len(races), "pair_samples": len(dataset), "features": len(feature_names)}))

    weights = train_logistic(dataset, feature_names, args.epochs, args.learning_rate, args.l2, args.seed)

    candidate = deepcopy(model)
    metadata = candidate.setdefault("metadata", {})
    metadata["pairwise_reranker"] = {
        "enabled": True,
        "weights": weights,
        "margin": args.margin,
        "rounds": args.rounds,
        "window": args.window,
        "max_gap": args.max_gap,
        "historical_races": len(races),
        "pair_samples": len(dataset),
        "seed": args.seed,
    }

    visible = evaluate_visible(repo_root, candidate)
    print(json.dumps({"visible_passed": visible, "baseline": baseline, "floor": args.min_keep_score}))

    if visible >= args.min_keep_score and visible > baseline:
        model_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        elite_dir = repo_root / "solution" / "elites"
        elite_dir.mkdir(parents=True, exist_ok=True)
        elite_name = f"model_params_pairwise_reranker_{visible:02d}_seed_{args.seed}.json"
        (elite_dir / elite_name).write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        print(json.dumps({"kept": True, "visible_passed": visible}))
    else:
        print(json.dumps({"kept": False, "visible_passed": visible}))


if __name__ == "__main__":
    main()
