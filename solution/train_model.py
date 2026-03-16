#!/usr/bin/env python3
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


REF_TEMP = 30.0
TRACKS = ("Monaco", "Monza", "Silverstone", "Suzuka", "Bahrain", "Spa", "COTA")


def _phase_bucket(lap_number, total_laps, progress_buckets):
    return min(progress_buckets - 1, ((lap_number - 1) * progress_buckets) // total_laps)


def _temp_bin(track_temp):
    t = int(round(float(track_temp)))
    return max(15, min(45, (t // 3) * 3))


def _ratio_bucket(value, bucket_count):
    clipped = max(0.0, min(0.999999, float(value)))
    return min(bucket_count - 1, int(clipped * bucket_count))


def _build_age_index_groups(feature_names):
    groups = {
        "lap": {"SOFT": [], "MEDIUM": [], "HARD": []},
        "temp": {"SOFT": [], "MEDIUM": [], "HARD": []},
    }
    for idx, name in enumerate(feature_names):
        if name.startswith("lap::") or name.startswith("temp::"):
            parts = name.split("::")
            if len(parts) != 3:
                continue
            kind, tire, age_str = parts
            if tire not in groups.get(kind, {}):
                continue
            try:
                age = int(age_str)
            except ValueError:
                continue
            groups[kind][tire].append((age, idx))

    ordered = {"lap": {}, "temp": {}}
    for kind in ("lap", "temp"):
        for tire in ("SOFT", "MEDIUM", "HARD"):
            ordered[kind][tire] = [idx for _age, idx in sorted(groups[kind][tire])]
    return ordered


def _apply_monotonic_smoothing(weights, age_groups, strength, rounds=1):
    if strength <= 0.0:
        return

    for _ in range(rounds):
        for kind in ("lap", "temp"):
            for tire in ("SOFT", "MEDIUM", "HARD"):
                idxs = age_groups[kind][tire]
                if len(idxs) < 2:
                    continue
                for i in range(len(idxs) - 1):
                    i0 = idxs[i]
                    i1 = idxs[i + 1]
                    violation = weights[i0] - weights[i1]
                    if violation > 0.0:
                        delta = 0.5 * strength * violation
                        weights[i0] -= delta
                        weights[i1] += delta


def _iter_laps(strategy, total_laps):
    tire = strategy["starting_tire"]
    stop_map = {int(stop["lap"]): stop["to_tire"] for stop in strategy.get("pit_stops", [])}
    tire_age = 0
    stint_index = 0

    for lap_number in range(1, total_laps + 1):
        tire_age += 1
        yield lap_number, tire, tire_age, stint_index
        if lap_number in stop_map:
            tire = stop_map[lap_number]
            tire_age = 0
            stint_index += 1


def extract_features(strategy, race_config, config):
    total_laps = int(race_config["total_laps"])
    temp_norm = (float(race_config["track_temp"]) - REF_TEMP) / config["temp_scale"]
    pit_stops = strategy.get("pit_stops", [])

    age_bucket_cap = config.get("age_bucket_cap", 50)
    progress_buckets = config.get("progress_buckets", 8)
    late_hinges = config.get("late_hinges", [14, 22, 30, 38])

    feats = {f"driver::{strategy['driver_id']}": 1.0}
    track = race_config["track"]
    tbin = _temp_bin(race_config["track_temp"])
    driver_id = strategy["driver_id"]
    feats[f"track::{track}"] = 1.0
    feats[f"track_temp::{track}"] = temp_norm
    feats[f"driver_track::{driver_id}::{track}"] = 1.0
    feats[f"driver_temp_bin::{driver_id}::{tbin}"] = 1.0

    for lap_number, tire, tire_age, stint_index in _iter_laps(strategy, total_laps):
        bucket = min(tire_age, age_bucket_cap)
        feats[f"lap::{tire}::{bucket}"] = feats.get(f"lap::{tire}::{bucket}", 0.0) + 1.0
        feats[f"temp::{tire}::{bucket}"] = feats.get(f"temp::{tire}::{bucket}", 0.0) + temp_norm

        stint_bucket = min(stint_index, 2)
        feats[f"stint::{stint_bucket}::{tire}"] = feats.get(f"stint::{stint_bucket}::{tire}", 0.0) + 1.0
        feats[f"stint_temp::{stint_bucket}::{tire}"] = feats.get(f"stint_temp::{stint_bucket}::{tire}", 0.0) + temp_norm

        for hinge in late_hinges:
            if tire_age > hinge:
                over = float(tire_age - hinge)
                feats[f"late::{tire}::{hinge}"] = feats.get(f"late::{tire}::{hinge}", 0.0) + over
                feats[f"late_temp::{tire}::{hinge}"] = feats.get(f"late_temp::{tire}::{hinge}", 0.0) + (over * temp_norm)

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

    feats["pit_count"] = float(len(pit_stops))
    feats["pit_lane_time"] = float(race_config["pit_lane_time"]) * len(pit_stops)

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
    dense_dataset = []
    all_feature_names = set()

    for race in races:
        race_config = race["race_config"]
        true_order = race["finishing_positions"]
        dense_map = {}

        for strategy in race["strategies"].values():
            driver_id = strategy["driver_id"]
            feats = extract_features(strategy, race_config, config)
            dense_map[driver_id] = feats
            all_feature_names.update(feats.keys())

        dense_dataset.append((true_order, dense_map))

    feature_names = sorted(all_feature_names)
    index_by_name = {name: idx for idx, name in enumerate(feature_names)}

    sparse_dataset = []
    for true_order, dense_map in dense_dataset:
        sparse_map = {}
        for driver_id, feats in dense_map.items():
            pairs = sorted((index_by_name[name], value) for name, value in feats.items())
            indices = [idx for idx, _ in pairs]
            values = [val for _, val in pairs]
            sparse_map[driver_id] = (indices, values)
        ordered_sparse = [sparse_map[driver_id] for driver_id in true_order]
        sparse_dataset.append((true_order, sparse_map, ordered_sparse))

    return sparse_dataset, feature_names


def score_sparse(weights, sparse_feats):
    indices, values = sparse_feats
    total = 0.0
    for idx, value in zip(indices, values):
        total += weights[idx] * value
    return total


def update_pair(weights, slow_sparse, fast_sparse, gradient, lr, l2):
    slow_indices, slow_values = slow_sparse
    fast_indices, fast_values = fast_sparse
    i = 0
    j = 0

    while i < len(slow_indices) and j < len(fast_indices):
        si = slow_indices[i]
        fi = fast_indices[j]
        if si == fi:
            delta = slow_values[i] - fast_values[j]
            weights[si] += lr * (gradient * delta - l2 * weights[si])
            i += 1
            j += 1
        elif si < fi:
            delta = slow_values[i]
            weights[si] += lr * (gradient * delta - l2 * weights[si])
            i += 1
        else:
            delta = -fast_values[j]
            weights[fi] += lr * (gradient * delta - l2 * weights[fi])
            j += 1

    while i < len(slow_indices):
        si = slow_indices[i]
        delta = slow_values[i]
        weights[si] += lr * (gradient * delta - l2 * weights[si])
        i += 1

    while j < len(fast_indices):
        fi = fast_indices[j]
        delta = -fast_values[j]
        weights[fi] += lr * (gradient * delta - l2 * weights[fi])
        j += 1


def evaluate(dataset, weights):
    exact = 0
    pair_ok = 0
    pair_total = 0

    for true_order, sparse_map, _ordered_sparse in dataset:
        pred = sorted(true_order, key=lambda driver_id: score_sparse(weights, sparse_map[driver_id]))
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


def infer_tie_breaker_scores(races):
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

    scores = {}
    for driver_id in [f"D{index:03d}" for index in range(1, 21)]:
        total = wins[driver_id] + losses[driver_id]
        scores[driver_id] = 0.0 if total == 0 else (wins[driver_id] - losses[driver_id]) / total
    return scores


def train_ridge_regression(train_races, config, feature_names, l2=1e-4, driver_prior=None, monotonic_strength=0.0):
    import numpy as np

    n_feat = len(feature_names)
    feat_idx = {name: i for i, name in enumerate(feature_names)}

    def feats_for_driver(strategy, race_config):
        feats = extract_features(strategy, race_config, config)
        vec = [0.0] * n_feat
        for name, val in feats.items():
            if name in feat_idx:
                vec[feat_idx[name]] = val
        return vec

    XtX = np.zeros((n_feat, n_feat), dtype=np.float64)
    Xty = np.zeros(n_feat, dtype=np.float64)

    for race in train_races:
        rc = race["race_config"]
        finish = race["finishing_positions"]
        n = len(finish)
        strat_by_driver = {s["driver_id"]: s for s in race["strategies"].values()}
        F = np.array([feats_for_driver(strat_by_driver[d], rc) for d in finish], dtype=np.float64)

        f = F.sum(axis=0)
        rank_w = np.arange(n, dtype=np.float64) * 2 - (n - 1)
        XtX += n * (F.T @ F) - np.outer(f, f)
        Xty += F.T @ rank_w

    if driver_prior:
        prior_vec = np.zeros(n_feat, dtype=np.float64)
        for name, val in driver_prior.items():
            idx = feat_idx.get(name)
            if idx is not None:
                prior_vec[idx] = val
        Xty += l2 * prior_vec

    XtX += l2 * np.eye(n_feat, dtype=np.float64)
    weights = np.linalg.solve(XtX, Xty).tolist()

    if monotonic_strength > 0.0:
        age_groups = _build_age_index_groups(feature_names)
        _apply_monotonic_smoothing(weights, age_groups, monotonic_strength, rounds=4)
    return weights


def train_pairwise_logistic(
    dataset,
    feature_names,
    epochs,
    pairs_per_race,
    lr,
    l2,
    seed,
    initial_weights=None,
    monotonic_strength=0.0,
    exact_focus=0.9,
    topk_weight=1.4,
):
    rng = random.Random(seed)
    weights = [0.0 for _ in feature_names]
    index_by_name = {name: idx for idx, name in enumerate(feature_names)}
    if initial_weights:
        for key, value in initial_weights.items():
            idx = index_by_name.get(key)
            if idx is not None:
                weights[idx] = value

    age_groups = _build_age_index_groups(feature_names) if monotonic_strength > 0.0 else None

    for epoch in range(epochs):
        rng.shuffle(dataset)
        avg_margin = 0.0
        count = 0

        for _true_order, _sparse_map, ordered_sparse in dataset:
            n = len(ordered_sparse)
            for _ in range(pairs_per_race):
                if rng.random() < 0.75:
                    i = rng.randint(0, n - 2)
                    j = min(n - 1, i + 1 + rng.randint(0, 4))
                else:
                    i = rng.randint(0, n - 2)
                    j = rng.randint(i + 1, n - 1)

                fast_sparse = ordered_sparse[i]
                slow_sparse = ordered_sparse[j]
                margin = score_sparse(weights, slow_sparse) - score_sparse(weights, fast_sparse)
                avg_margin += margin
                count += 1

                if margin >= 0:
                    exp_term = math.exp(-margin)
                    gradient = exp_term / (1.0 + exp_term)
                else:
                    exp_term = math.exp(margin)
                    gradient = 1.0 / (1.0 + exp_term)

                focus = 1.0 + exact_focus * math.exp(-abs(margin))
                if i < 10 or j < 10:
                    focus *= topk_weight
                gradient *= min(6.0, focus)

                update_pair(weights, slow_sparse, fast_sparse, gradient, lr, l2)

        print(f"epoch={epoch + 1} pairs={count} avg_margin={avg_margin / max(1, count):.6f}")

        if age_groups is not None:
            _apply_monotonic_smoothing(weights, age_groups, monotonic_strength, rounds=1)

    return weights


def main():
    parser = argparse.ArgumentParser(description="Train mechanistic v2 lap-by-lap model from historical races")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--pairs-per-race", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.4)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--temp-scale", type=float, default=15.0)
    parser.add_argument("--age-bucket-cap", type=int, default=50)
    parser.add_argument("--progress-buckets", type=int, default=8)
    parser.add_argument("--late-hinges", type=str, default="14,22,30,38")
    parser.add_argument("--monotonic-strength", type=float, default=0.18)
    parser.add_argument("--exact-focus", type=float, default=0.9)
    parser.add_argument("--topk-weight", type=float, default=1.4)
    parser.add_argument("--tie-gap-threshold", type=float, default=0.06)
    args = parser.parse_args()

    late_hinges = [int(x.strip()) for x in args.late_hinges.split(",") if x.strip()]

    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "temp_scale": args.temp_scale,
        "age_bucket_cap": args.age_bucket_cap,
        "progress_buckets": args.progress_buckets,
        "late_hinges": late_hinges,
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
    tie_break_scores = infer_tie_breaker_scores(train_races)

    print(f"races={len(races)} train={len(train_data)} val={len(val_data)} features={len(feature_names)}")
    print("inferred_driver_order=" + ",".join(inferred_driver_order))

    if NUMPY_AVAILABLE:
        print("Training via analytical ridge regression (numpy)...")
        weights = train_ridge_regression(
            train_races,
            config,
            feature_names,
            l2=args.l2,
            driver_prior=driver_prior,
            monotonic_strength=args.monotonic_strength,
        )
        if args.epochs > 0:
            print(f"Fine-tuning with {args.epochs} SGD epochs...")
            weights = train_pairwise_logistic(
                train_data,
                feature_names,
                epochs=args.epochs,
                pairs_per_race=args.pairs_per_race,
                lr=args.learning_rate * 0.1,
                l2=args.l2,
                seed=args.seed,
                initial_weights={name: weights[i] for i, name in enumerate(feature_names)},
                monotonic_strength=args.monotonic_strength,
                exact_focus=args.exact_focus,
                topk_weight=args.topk_weight,
            )
    else:
        print("numpy not available, falling back to pairwise logistic SGD...")
        weights = train_pairwise_logistic(
            train_data,
            feature_names,
            epochs=args.epochs,
            pairs_per_race=args.pairs_per_race,
            lr=args.learning_rate,
            l2=args.l2,
            seed=args.seed,
            initial_weights=driver_prior,
            monotonic_strength=args.monotonic_strength,
            exact_focus=args.exact_focus,
            topk_weight=args.topk_weight,
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

    feature_weights = {name: weights[idx] for idx, name in enumerate(feature_names)}

    model = {
        "feature_weights": feature_weights,
        "mechanistic_config": config,
        "metadata": {
            "model": "mechanistic_v2_highres",
            "trained_on_races": len(races),
            "train_races": len(train_races),
            "val_races": len(val_races),
            "epochs": args.epochs,
            "pairs_per_race": args.pairs_per_race,
            "seed": args.seed,
            "monotonic_strength": args.monotonic_strength,
            "exact_focus": args.exact_focus,
            "topk_weight": args.topk_weight,
            "tie_gap_threshold": args.tie_gap_threshold,
            "late_hinges": late_hinges,
            "train_pair_rate": train_metrics["pair_rate"],
            "val_pair_rate": val_metrics["pair_rate"],
            "train_exact_rate": train_metrics["exact_rate"],
            "val_exact_rate": val_metrics["exact_rate"],
            "inferred_driver_order": inferred_driver_order,
            "tie_break_scores": tie_break_scores,
        },
    }

    out_path = Path(__file__).with_name("model_params.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    print(f"wrote_model={out_path}")


if __name__ == "__main__":
    main()
