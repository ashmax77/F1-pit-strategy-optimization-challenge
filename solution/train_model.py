#!/usr/bin/env python3
import argparse
import json
import math
import random
from pathlib import Path


TIRES = ("SOFT", "MEDIUM", "HARD")
TRACKS = ("Bahrain", "COTA", "Monaco", "Monza", "Silverstone", "Spa", "Suzuka")
REF_TEMP = 30.0


def _compound_stats(starting_tire, pit_stops, total_laps):
    laps_by_tire = {t: 0 for t in TIRES}
    age_sum_by_tire = {t: 0.0 for t in TIRES}
    age2_sum_by_tire = {t: 0.0 for t in TIRES}
    stints_by_tire = {t: 0 for t in TIRES}
    max_stint_by_tire = {t: 0 for t in TIRES}

    current_tire = starting_tire
    prev_lap = 0

    for stop in pit_stops:
        stop_lap = int(stop["lap"])
        stint_len = stop_lap - prev_lap
        if stint_len > 0:
            laps_by_tire[current_tire] += stint_len
            age_sum_by_tire[current_tire] += (stint_len * (stint_len + 1)) / 2.0
            age2_sum_by_tire[current_tire] += (
                stint_len * (stint_len + 1) * (2 * stint_len + 1)
            ) / 6.0
            stints_by_tire[current_tire] += 1
            max_stint_by_tire[current_tire] = max(max_stint_by_tire[current_tire], stint_len)
        current_tire = stop["to_tire"]
        prev_lap = stop_lap

    final_stint_len = total_laps - prev_lap
    if final_stint_len > 0:
        laps_by_tire[current_tire] += final_stint_len
        age_sum_by_tire[current_tire] += (final_stint_len * (final_stint_len + 1)) / 2.0
        age2_sum_by_tire[current_tire] += (
            final_stint_len * (final_stint_len + 1) * (2 * final_stint_len + 1)
        ) / 6.0
        stints_by_tire[current_tire] += 1
        max_stint_by_tire[current_tire] = max(max_stint_by_tire[current_tire], final_stint_len)

    return laps_by_tire, age_sum_by_tire, age2_sum_by_tire, stints_by_tire, max_stint_by_tire


def extract_features(strategy, race_config, scales):
    total_laps = int(race_config["total_laps"])
    track = race_config["track"]
    track_temp = float(race_config["track_temp"])
    temp_norm = (track_temp - REF_TEMP) / scales["temp_norm"]
    pit_lane_time = float(race_config["pit_lane_time"])
    pit_stops = strategy.get("pit_stops", [])

    (
        laps_by_tire,
        age_sum_by_tire,
        age2_sum_by_tire,
        stints_by_tire,
        max_stint_by_tire,
    ) = _compound_stats(strategy["starting_tire"], pit_stops, total_laps)

    feats = {
        "stops": len(pit_stops) / scales["stops"],
        "pit_time_stops": (pit_lane_time * len(pit_stops)) / scales["pit_time_stops"],
        f"drv_{strategy['driver_id']}": 1.0,
    }

    for tire in TIRES:
        key = tire.lower()
        laps = laps_by_tire[tire] / scales["laps"]
        ages = age_sum_by_tire[tire] / scales["ages"]
        ages2 = age2_sum_by_tire[tire] / scales["ages2"]
        stints = stints_by_tire[tire] / scales["stops"]
        max_stint = max_stint_by_tire[tire] / scales["max_stint"]

        feats[f"laps_{key}"] = laps
        feats[f"ages_{key}"] = ages
        feats[f"ages2_{key}"] = ages2
        feats[f"stints_{key}"] = stints
        feats[f"maxstint_{key}"] = max_stint

        feats[f"temp_laps_{key}"] = temp_norm * laps
        feats[f"temp_ages_{key}"] = temp_norm * ages
        feats[f"temp_ages2_{key}"] = temp_norm * ages2

        if track in TRACKS:
            feats[f"trk_{track}_laps_{key}"] = laps
            feats[f"trk_{track}_ages_{key}"] = ages

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


def prepare_race_dataset(races, scales):
    dataset = []
    all_feature_names = set()

    for race in races:
        race_config = race["race_config"]
        strategies = race["strategies"]
        true_order = race["finishing_positions"]

        feature_map = {}
        for strategy in strategies.values():
            driver_id = strategy["driver_id"]
            feats = extract_features(strategy, race_config, scales)
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
        pred = sorted(
            true_order,
            key=lambda d: score_driver(weights, feature_map[d]),
        )
        if pred == true_order:
            exact += 1

        rank = {d: i for i, d in enumerate(true_order)}
        n = len(pred)
        for i in range(n):
            for j in range(i + 1, n):
                if rank[pred[i]] < rank[pred[j]]:
                    pair_ok += 1
                pair_total += 1

    return {
        "exact": exact,
        "total": len(dataset),
        "exact_rate": (100.0 * exact / max(1, len(dataset))),
        "pair_rate": (100.0 * pair_ok / max(1, pair_total)),
    }


def train_pairwise_logistic(dataset, feature_names, epochs, pairs_per_race, lr, l2, seed):
    rng = random.Random(seed)
    weights = {name: 0.0 for name in feature_names}

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
                diff = {k: slow_feats.get(k, 0.0) - fast_feats.get(k, 0.0) for k in keys}

                margin = dot(weights, diff)
                avg_margin += margin
                count += 1

                # Numerical-stable sigmoid(-margin)
                if margin >= 0:
                    exp_term = math.exp(-margin)
                    g = exp_term / (1.0 + exp_term)
                else:
                    exp_term = math.exp(margin)
                    g = 1.0 / (1.0 + exp_term)

                for key, value in diff.items():
                    weights[key] = weights.get(key, 0.0) + lr * (g * value - l2 * weights.get(key, 0.0))

        print(f"epoch={epoch + 1} pairs={count} avg_margin={avg_margin / max(1, count):.6f}")

    return weights


def main():
    parser = argparse.ArgumentParser(description="Train weighted ranking model from historical races")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--pairs-per-race", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--l2", type=float, default=2e-4)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scales = {
        "laps": 100.0,
        "ages": 5000.0,
        "ages2": 200000.0,
        "stops": 6.0,
        "pit_time_stops": 180.0,
        "max_stint": 70.0,
        "temp_norm": 15.0,
    }

    races = load_historical_races(repo_root, args.max_files)
    random.Random(args.seed).shuffle(races)

    split = int(0.85 * len(races))
    train_races = races[:split]
    val_races = races[split:]

    train_data, feat_names_train = prepare_race_dataset(train_races, scales)
    val_data, feat_names_val = prepare_race_dataset(val_races, scales)

    feature_names = sorted(set(feat_names_train).union(feat_names_val))
    print(f"races={len(races)} train={len(train_data)} val={len(val_data)} features={len(feature_names)}")

    weights = train_pairwise_logistic(
        train_data,
        feature_names,
        epochs=args.epochs,
        pairs_per_race=args.pairs_per_race,
        lr=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
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
        "feature_scales": scales,
        "metadata": {
            "model": "pairwise_logistic_sparse_v2",
            "trained_on_races": len(races),
            "train_races": len(train_races),
            "val_races": len(val_races),
            "epochs": args.epochs,
            "pairs_per_race": args.pairs_per_race,
            "seed": args.seed,
            "train_pair_rate": train_metrics["pair_rate"],
            "val_pair_rate": val_metrics["pair_rate"],
        },
    }

    out_path = Path(__file__).with_name("model_params.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    print(f"wrote_model={out_path}")


if __name__ == "__main__":
    main()
