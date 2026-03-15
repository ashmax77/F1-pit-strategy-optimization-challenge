#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


REF_TEMP = 30.0


def _age_bucket(age, age_bucket_cap):
    return min(age, age_bucket_cap)


def _phase_bucket(lap_number, total_laps, progress_buckets):
    return min(progress_buckets - 1, ((lap_number - 1) * progress_buckets) // total_laps)


def _iter_laps(strategy, total_laps):
    tire = strategy["starting_tire"]
    stop_map = {int(stop["lap"]): stop["to_tire"] for stop in strategy.get("pit_stops", [])}
    tire_age = 0

    for lap_number in range(1, total_laps + 1):
        tire_age += 1
        yield lap_number, tire, tire_age
        if lap_number in stop_map:
            tire = stop_map[lap_number]
            tire_age = 0


def extract_features(strategy, race_config, config):
    total_laps = int(race_config["total_laps"])
    temp_norm = (float(race_config["track_temp"]) - REF_TEMP) / config["temp_scale"]
    pit_stops = strategy.get("pit_stops", [])

    feats = {
        f"driver::{strategy['driver_id']}": 1.0,
        "pit_count": float(len(pit_stops)),
        "pit_lane_time": float(race_config["pit_lane_time"]) * len(pit_stops),
    }

    for lap_number, tire, tire_age in _iter_laps(strategy, total_laps):
        age_bucket = _age_bucket(tire_age, config["age_bucket_cap"])
        phase_bucket = _phase_bucket(lap_number, total_laps, config["progress_buckets"])
        lap_key = f"lap::{tire}::{age_bucket}"
        temp_key = f"temp::{tire}::{age_bucket}"
        phase_key = f"phase::{tire}::{phase_bucket}"
        feats[lap_key] = feats.get(lap_key, 0.0) + 1.0
        feats[temp_key] = feats.get(temp_key, 0.0) + temp_norm
        feats[phase_key] = feats.get(phase_key, 0.0) + 1.0

    return feats


def load_historical_races(repo_root, max_files=None):
    hist_dir = repo_root / "data" / "historical_races"
    files = sorted(hist_dir.glob("races_*.json"))
    if max_files is not None:
        files = files[:max_files]

    races = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as fh:
            races.extend(json.load(fh))
    return races


def prepare_race_dataset(races, config):
    dataset = []
    all_feature_names = set()

    for race in races:
        race_config = race["race_config"]
        true_order = race["finishing_positions"]
        feature_map = {}

        for strategy in race["strategies"].values():
            driver_id = strategy["driver_id"]
            feats = extract_features(strategy, race_config, config)
            feature_map[driver_id] = feats
            all_feature_names.update(feats.keys())

        dataset.append((true_order, feature_map))

    return dataset, sorted(all_feature_names)


def dot(weights, diff):
    total = 0.0
    for key, value in diff.items():
        total += weights.get(key, 0.0) * value
    return total


def score_driver(weights, features):
    total = 0.0
    for key, value in features.items():
        total += weights.get(key, 0.0) * value
    return total


def evaluate(dataset, weights):
    exact = 0
    pair_ok = 0
    pair_total = 0

    for true_order, feature_map in dataset:
        pred = sorted(true_order, key=lambda driver_id: score_driver(weights, feature_map[driver_id]))
        if pred == true_order:
            exact += 1

        rank = {driver_id: index for index, driver_id in enumerate(true_order)}
        for i in range(len(pred)):
            for j in range(i + 1, len(pred)):
                if rank[pred[i]] < rank[pred[j]]:
                    pair_ok += 1
                pair_total += 1

    return {
        "exact": exact,
        "total": len(dataset),
        "exact_rate": (100.0 * exact / max(1, len(dataset))),
        "pair_rate": (100.0 * pair_ok / max(1, pair_total)),
    }


def _strategy_signature(strategy):
    return (
        strategy["starting_tire"],
        tuple((int(stop["lap"]), stop["to_tire"]) for stop in strategy.get("pit_stops", [])),
    )


def infer_driver_prior(races):
    wins = Counter()
    losses = Counter()

    for race in races:
        by_signature = defaultdict(list)
        finish_index = {driver_id: idx for idx, driver_id in enumerate(race["finishing_positions"])}

        for strategy in race["strategies"].values():
            by_signature[_strategy_signature(strategy)].append(strategy["driver_id"])

        for drivers in by_signature.values():
            if len(drivers) < 2:
                continue
            ordered = sorted(drivers, key=lambda driver_id: finish_index[driver_id])
            for idx, winner in enumerate(ordered):
                wins[winner] += len(ordered) - idx - 1
                losses[winner] += idx

    ranked = sorted(
        {f"D{index:03d}" for index in range(1, 21)},
        key=lambda driver_id: (wins[driver_id] - losses[driver_id], wins[driver_id]),
        reverse=True,
    )
    center = (len(ranked) - 1) / 2.0
    scale = max(1.0, center)

    prior = {}
    for idx, driver_id in enumerate(ranked):
        prior[f"driver::{driver_id}"] = (idx - center) / scale
    return prior, ranked


def train_pairwise_logistic(dataset, feature_names, epochs, pairs_per_race, lr, l2, seed, initial_weights=None):
    rng = random.Random(seed)
    weights = {name: 0.0 for name in feature_names}
    if initial_weights:
        for key, value in initial_weights.items():
            if key in weights:
                weights[key] = value

    for epoch in range(epochs):
        rng.shuffle(dataset)
        avg_margin = 0.0
        count = 0

        for true_order, feature_map in dataset:
            n = len(true_order)
            for _ in range(pairs_per_race):
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
                fast = true_order[i]
                slow = true_order[j]

                slow_feats = feature_map[slow]
                fast_feats = feature_map[fast]

                keys = set(slow_feats.keys())
                keys.update(fast_feats.keys())
                diff = {key: slow_feats.get(key, 0.0) - fast_feats.get(key, 0.0) for key in keys}

                margin = dot(weights, diff)
                avg_margin += margin
                count += 1

                if margin >= 0:
                    exp_term = math.exp(-margin)
                    gradient = exp_term / (1.0 + exp_term)
                else:
                    exp_term = math.exp(margin)
                    gradient = 1.0 / (1.0 + exp_term)

                for key, value in diff.items():
                    weights[key] = weights.get(key, 0.0) + lr * (gradient * value - l2 * weights.get(key, 0.0))

        print(f"epoch={epoch + 1} pairs={count} avg_margin={avg_margin / max(1, count):.6f}")

    return weights


def main():
    parser = argparse.ArgumentParser(description="Train mechanistic lap-by-lap timing model from historical races")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--pairs-per-race", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--age-bucket-cap", type=int, default=40)
    parser.add_argument("--progress-buckets", type=int, default=8)
    parser.add_argument("--temp-scale", type=float, default=15.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "age_bucket_cap": args.age_bucket_cap,
        "progress_buckets": args.progress_buckets,
        "temp_scale": args.temp_scale,
    }

    races = load_historical_races(repo_root, args.max_files)
    random.Random(args.seed).shuffle(races)

    split = int(0.85 * len(races))
    train_races = races[:split]
    val_races = races[split:]

    train_data, feat_names_train = prepare_race_dataset(train_races, config)
    val_data, feat_names_val = prepare_race_dataset(val_races, config)
    feature_names = sorted(set(feat_names_train).union(feat_names_val))
    driver_prior, inferred_driver_order = infer_driver_prior(train_races)

    print(f"races={len(races)} train={len(train_data)} val={len(val_data)} features={len(feature_names)}")
    print("inferred_driver_order=" + ",".join(inferred_driver_order))

    weights = train_pairwise_logistic(
        train_data,
        feature_names,
        epochs=args.epochs,
        pairs_per_race=args.pairs_per_race,
        lr=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
        initial_weights=driver_prior,
    )

    train_metrics = evaluate(train_data, weights)
    val_metrics = evaluate(val_data, weights)
    print(
        "train_exact={exact}/{total} ({rate:.2f}%) pairwise={pair:.2f}%".format(
            exact=train_metrics["exact"],
            total=train_metrics["total"],
            rate=train_metrics["exact_rate"],
            pair=train_metrics["pair_rate"],
        )
    )
    print(
        "val_exact={exact}/{total} ({rate:.2f}%) pairwise={pair:.2f}%".format(
            exact=val_metrics["exact"],
            total=val_metrics["total"],
            rate=val_metrics["exact_rate"],
            pair=val_metrics["pair_rate"],
        )
    )

    model = {
        "feature_weights": weights,
        "mechanistic_config": config,
        "metadata": {
            "model": "pairwise_logistic_mechanistic_v4",
            "trained_on_races": len(races),
            "train_races": len(train_races),
            "val_races": len(val_races),
            "epochs": args.epochs,
            "pairs_per_race": args.pairs_per_race,
            "seed": args.seed,
            "train_pair_rate": train_metrics["pair_rate"],
            "val_pair_rate": val_metrics["pair_rate"],
            "train_exact_rate": train_metrics["exact_rate"],
            "val_exact_rate": val_metrics["exact_rate"],
            "inferred_driver_order": inferred_driver_order,
        },
    }

    out_path = Path(__file__).with_name("model_params.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    print(f"wrote_model={out_path}")


if __name__ == "__main__":
    main()
