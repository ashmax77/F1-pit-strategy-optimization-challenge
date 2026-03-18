#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REF_TEMP = 30.0
COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


def _phase_bucket(lap_number, total_laps, progress_buckets):
    return min(progress_buckets - 1, ((lap_number - 1) * progress_buckets) // total_laps)


def _ratio_bucket(value, bucket_count):
    clipped = max(0.0, min(0.999999, float(value)))
    return min(bucket_count - 1, int(clipped * bucket_count))


def _temp_bin(track_temp):
    t = int(round(float(track_temp)))
    return max(15, min(45, (t // 3) * 3))


def _temp_regime(track_temp):
    temp = float(track_temp)
    if temp >= 33.0:
        return "hot"
    if temp <= 27.0:
        return "cold"
    return "mid"


def _driver_rank_score(driver_id):
    # Hidden baseline from data is stable: lower driver number tends to be faster.
    try:
        rank = int(str(driver_id)[1:])
    except (ValueError, TypeError):
        rank = 10
    rank = max(1, min(20, rank))
    return float(rank - 1) / 19.0


def _strategy_signature(strategy):
    stops = strategy.get("pit_stops", [])
    compact = tuple((int(stop["lap"]), stop["to_tire"]) for stop in stops)
    return (strategy.get("starting_tire", ""), compact)


def _default_model():
    return {
        "feature_weights": {},
        "mechanistic_config": {
            "temp_scale": 15.0,
            "age_bucket_cap": 50,
        },
        "metadata": {"source": "default"},
    }


def _load_model():
    model_path = Path(__file__).with_name("model_params.json")
    if not model_path.exists():
        return _default_model()
    with model_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _driver_relative_time(strategy, race_config, model):
    weights = model.get("feature_weights", {})
    config = model.get("mechanistic_config", {})
    temp_scale = float(config.get("temp_scale", 15.0))
    age_bucket_cap = int(config.get("age_bucket_cap", 50))
    progress_buckets = int(config.get("progress_buckets", 8))
    late_hinges = [int(h) for h in config.get("late_hinges", [14, 22, 30, 38])]

    total_laps = int(race_config["total_laps"])
    track = race_config.get("track", "")
    track_temp = float(race_config["track_temp"])
    pit_lane_time = float(race_config["pit_lane_time"])
    pit_stops = strategy.get("pit_stops", [])
    temp_norm = (track_temp - REF_TEMP) / temp_scale
    tbin = _temp_bin(track_temp)

    stop_map = {int(stop["lap"]): stop["to_tire"] for stop in pit_stops}
    driver_id = strategy["driver_id"]

    relative_time = weights.get(f"driver::{driver_id}", 0.0)
    relative_time += weights.get("pit_count", 0.0) * len(pit_stops)
    relative_time += weights.get("pit_lane_time", 0.0) * (pit_lane_time * len(pit_stops))

    # Track and driver-track features
    relative_time += weights.get(f"track::{track}", 0.0)
    relative_time += weights.get(f"track_temp::{track}", 0.0) * temp_norm
    relative_time += weights.get(f"driver_track::{driver_id}::{track}", 0.0)
    relative_time += weights.get(f"driver_temp_bin::{driver_id}::{tbin}", 0.0)

    for stop in pit_stops:
        phase_bucket = _phase_bucket(int(stop["lap"]), total_laps, progress_buckets)
        relative_time += weights.get(f"pit_phase::{phase_bucket}", 0.0)

    last_stop_lap = max([int(stop["lap"]) for stop in pit_stops] + [0])
    last_tire = pit_stops[-1]["to_tire"] if pit_stops else strategy["starting_tire"]
    last_stint_len = total_laps - last_stop_lap
    last_stint_phase = _phase_bucket(max(1, last_stop_lap + 1), total_laps, progress_buckets)
    last_stop_ratio = float(last_stop_lap) / max(1, total_laps)
    last_stint_ratio = float(last_stint_len) / max(1, total_laps)
    last_stop_bucket = _ratio_bucket(last_stop_ratio, 10)
    last_stint_bucket = _ratio_bucket(last_stint_ratio, 10)

    relative_time += weights.get("last_stint_len", 0.0) * last_stint_len
    relative_time += weights.get(f"last_stint_tire::{last_tire}", 0.0)
    relative_time += weights.get(f"last_stint_phase::{last_stint_phase}", 0.0)
    relative_time += weights.get(f"last_stint_temp::{last_tire}", 0.0) * (temp_norm * last_stint_len)

    # Late-stop and final-stint bucket features
    relative_time += weights.get(f"last_stop_bin::{last_stop_bucket}", 0.0)
    relative_time += weights.get(f"last_stint_bin::{last_stint_bucket}", 0.0)
    relative_time += weights.get(f"final_tire_track::{track}::{last_tire}", 0.0)
    relative_time += weights.get(f"final_tire_stopbin::{last_tire}::{last_stop_bucket}", 0.0)
    relative_time += weights.get(f"track_last_stop::{track}::{last_stop_bucket}", 0.0)
    relative_time += weights.get(f"temp_last_stop::{tbin}::{last_stop_bucket}", 0.0)
    relative_time += weights.get(f"temp_final_tire::{tbin}::{last_tire}", 0.0)
    relative_time += weights.get(f"track_temp_stop::{track}::{tbin}::{last_stop_bucket}", 0.0)

    current_tire = strategy["starting_tire"]
    tire_age = 0
    stint_index = 0
    for lap_number in range(1, total_laps + 1):
        tire_age += 1
        bucket = min(tire_age, age_bucket_cap)
        relative_time += weights.get(f"lap::{current_tire}::{bucket}", 0.0)
        relative_time += weights.get(f"temp::{current_tire}::{bucket}", 0.0) * temp_norm
        stint_bucket = min(stint_index, 2)
        relative_time += weights.get(f"stint::{stint_bucket}::{current_tire}", 0.0)
        relative_time += weights.get(f"stint_temp::{stint_bucket}::{current_tire}", 0.0) * temp_norm
        # Hinge-based late-stint degradation features (cumulative excess above each hinge)
        for hinge in late_hinges:
            if tire_age > hinge:
                over = float(tire_age - hinge)
                relative_time += weights.get(f"late::{current_tire}::{hinge}", 0.0) * over
                relative_time += weights.get(f"late_temp::{current_tire}::{hinge}", 0.0) * (over * temp_norm)
        if lap_number in stop_map:
            current_tire = stop_map[lap_number]
            tire_age = 0
            stint_index += 1

    return relative_time


def _driver_relative_time_legacy(strategy, race_config, model):
    # Backward-compatible alias used by phase6-8 helper scripts.
    return _driver_relative_time(strategy, race_config, model)


def _apply_runtime_adjustments(relative_time, strategy, race_config, model):
    """Apply lightweight metadata-driven corrections used by phase5 tuner."""
    metadata = model.get("metadata", {})
    total_laps = int(race_config["total_laps"])
    stops = strategy.get("pit_stops", [])
    last_stop_lap = max([int(stop["lap"]) for stop in stops] + [0])
    last_tire = stops[-1]["to_tire"] if stops else strategy["starting_tire"]
    last_stint_len = total_laps - last_stop_lap
    last_stop_ratio = float(last_stop_lap) / max(1, total_laps)
    last_stint_ratio = float(last_stint_len) / max(1, total_laps)

    late_stop_hinge = float(metadata.get("late_stop_hinge", 0.7))
    early_stop_hinge = float(metadata.get("early_stop_hinge", 0.1))
    short_last_stint_hinge = float(metadata.get("short_last_stint_hinge", 0.14))
    long_last_stint_hinge = float(metadata.get("long_last_stint_hinge", 0.56))

    if last_stop_ratio > late_stop_hinge:
        relative_time += float(metadata.get("late_stop_penalty", 0.0)) * (last_stop_ratio - late_stop_hinge)
    if last_stop_ratio < early_stop_hinge:
        relative_time += float(metadata.get("early_stop_penalty", 0.0)) * (early_stop_hinge - last_stop_ratio)
    if last_stint_ratio < short_last_stint_hinge:
        relative_time += float(metadata.get("short_last_stint_penalty", 0.0)) * (short_last_stint_hinge - last_stint_ratio)
    if last_stint_ratio > long_last_stint_hinge:
        relative_time += float(metadata.get("long_last_stint_penalty", 0.0)) * (last_stint_ratio - long_last_stint_hinge)

    relative_time += float(metadata.get(f"final_tire_bias::{last_tire}", 0.0))
    return relative_time


def _dynamic_rerank_overrides(scored, metadata):
    if not metadata.get("rerank_dynamic_enabled", True):
        return {}
    if len(scored) < 3:
        return {}

    top_n = max(3, int(metadata.get("rerank_dynamic_top_n", 10)))
    gap_threshold = float(metadata.get("rerank_dynamic_gap_threshold", 0.95))
    min_close_pairs = max(1, int(metadata.get("rerank_dynamic_min_close_pairs", 4)))

    end = min(len(scored), top_n)
    close_pairs = 0
    for idx in range(end - 1):
        if (scored[idx + 1][0] - scored[idx][0]) <= gap_threshold:
            close_pairs += 1

    if close_pairs < min_close_pairs:
        return {}

    return {
        "pairwise_rounds_extra": max(0, int(metadata.get("pairwise_rounds_extra", 1))),
        "pairwise_window_extra": max(0, int(metadata.get("pairwise_window_extra", 1))),
        "pairwise_margin_boost": max(0.0, float(metadata.get("pairwise_margin_boost", 0.3))),
        "local_rounds_extra": max(0, int(metadata.get("rerank_rounds_extra", 1))),
        "local_window_extra": max(0, int(metadata.get("rerank_window_extra", 1))),
        "local_margin_boost": max(0.0, float(metadata.get("rerank_margin_boost", 0.2))),
    }


def _build_rerank_context(strategy, race_config):
    total_laps = int(race_config["total_laps"])
    track = race_config.get("track", "")
    regime = _temp_regime(race_config.get("track_temp", REF_TEMP))
    stops = strategy.get("pit_stops", [])
    last_stop_lap = max([int(stop["lap"]) for stop in stops] + [0])
    last_tire = stops[-1]["to_tire"] if stops else strategy["starting_tire"]
    last_stint_len = total_laps - last_stop_lap
    last_stop_ratio = float(last_stop_lap) / max(1, total_laps)
    last_stint_ratio = float(last_stint_len) / max(1, total_laps)
    return {
        "driver_id": strategy["driver_id"],
        "driver_rank_norm": _driver_rank_score(strategy["driver_id"]),
        "track": track,
        "regime": regime,
        "pit_count": len(stops),
        "last_tire": last_tire,
        "last_stop_ratio": last_stop_ratio,
        "last_stint_ratio": last_stint_ratio,
        "signature": _strategy_signature(strategy),
    }


def _pair_key(right_tire, left_tire):
    mapping = {
        ("HARD", "MEDIUM"): "hard_over_medium",
        ("HARD", "SOFT"): "hard_over_soft",
        ("SOFT", "MEDIUM"): "soft_over_medium",
        ("MEDIUM", "SOFT"): "medium_over_soft",
    }
    return mapping.get((right_tire, left_tire), "other")


def _pairwise_feature_map(left_context, right_context, gap):
    track = right_context["track"]
    regime = right_context["regime"]
    right_tire = right_context["last_tire"]
    left_tire = left_context["last_tire"]
    pair_key = _pair_key(right_tire, left_tire)

    features = {
        "bias": 1.0,
        "gap": float(gap),
        "rank_diff": right_context["driver_rank_norm"] - left_context["driver_rank_norm"],
        "last_stop_diff": right_context["last_stop_ratio"] - left_context["last_stop_ratio"],
        "last_stint_diff": right_context["last_stint_ratio"] - left_context["last_stint_ratio"],
        "pit_count_diff": float(right_context["pit_count"] - left_context["pit_count"]),
        "same_strategy": 1.0 if right_context["signature"] == left_context["signature"] else 0.0,
        f"track::{track}": 1.0,
        f"track_pair::{track}::{right_tire}::{left_tire}": 1.0,
        f"track_right_final::{track}::{right_tire}": 1.0,
    }
    if pair_key != "other":
        features[f"regime_pair::{regime}::{right_tire}::{left_tire}"] = 1.0
    return features


def _score_pairwise_model(features, weight_map):
    return sum(float(weight_map.get(name, 0.0)) * value for name, value in features.items())


def _apply_pairwise_reranker(scored, strategies, race_config, model, dynamic_overrides=None):
    metadata = model.get("metadata", {})
    reranker = metadata.get("pairwise_reranker")
    if not isinstance(reranker, dict) or not reranker.get("enabled", False):
        return scored

    weights = reranker.get("weights", {})
    if not weights:
        return scored

    margin = float(reranker.get("margin", 1.75))
    rounds = max(1, int(reranker.get("rounds", 2)))
    window = max(1, int(reranker.get("window", 3)))
    if dynamic_overrides:
        margin += float(dynamic_overrides.get("pairwise_margin_boost", 0.0))
        rounds += int(dynamic_overrides.get("pairwise_rounds_extra", 0))
        window += int(dynamic_overrides.get("pairwise_window_extra", 0))
    by_driver = {strategy["driver_id"]: strategy for strategy in strategies.values()}
    contexts = {driver_id: _build_rerank_context(strategy, race_config) for driver_id, strategy in by_driver.items()}

    out = list(scored)
    for _ in range(rounds):
        changed = False
        for i in range(len(out) - 1):
            max_j = min(len(out), i + 1 + window)
            for j in range(i + 1, max_j):
                left_t, left_d = out[j - 1]
                right_t, right_d = out[j]
                gap = right_t - left_t
                if gap > margin:
                    break
                feats = _pairwise_feature_map(contexts[left_d], contexts[right_d], gap)
                pref = _score_pairwise_model(feats, weights)
                if pref > 0.0:
                    out[j - 1], out[j] = out[j], out[j - 1]
                    changed = True
        if not changed:
            break
    return out


def _local_pair_preference(left_context, right_context, gap, metadata):
    pref = 0.0

    pref += float(metadata.get("rerank_pair_driver_rank_diff_weight", 0.0)) * (
        left_context["driver_rank_norm"] - right_context["driver_rank_norm"]
    )
    pref += float(metadata.get("rerank_pair_last_stop_diff_weight", 0.0)) * (
        right_context["last_stop_ratio"] - left_context["last_stop_ratio"]
    )
    pref += float(metadata.get("rerank_pair_last_stint_diff_weight", 0.0)) * (
        right_context["last_stint_ratio"] - left_context["last_stint_ratio"]
    )
    pref += float(metadata.get("rerank_gap_weight", 0.0)) * max(0.0, 1.0 - gap / max(1e-6, float(metadata.get("rerank_margin", 1.0))))

    right_tire = right_context["last_tire"]
    left_tire = left_context["last_tire"]
    regime = right_context["regime"]
    track = right_context["track"]

    pref += float(metadata.get(f"rerank_final_{right_tire.lower()}_weight", 0.0))
    pref += float(metadata.get(f"rerank_track_{track}_weight", 0.0))
    pref += float(metadata.get(f"rerank_track_tire_{track}_{right_tire}_weight", 0.0))
    pref += float(metadata.get(f"rerank_{regime}_{right_tire.lower()}_weight", 0.0))
    pref += float(metadata.get(f"rerank_pair_track_{track}_weight", 0.0))

    pair_key = _pair_key(right_tire, left_tire)
    if pair_key == "hard_over_medium":
        pref += float(metadata.get("rerank_pair_hard_over_soft_weight", 0.0)) * 0.25
    elif pair_key == "hard_over_soft":
        pref += float(metadata.get("rerank_pair_hard_over_soft_weight", 0.0))
        if regime == "cold":
            pref += float(metadata.get("rerank_pair_cold_hard_over_soft_weight", 0.0))
    elif pair_key == "soft_over_medium":
        pref += float(metadata.get("rerank_pair_soft_over_medium_weight", 0.0))
    elif pair_key == "medium_over_soft":
        pref += float(metadata.get("rerank_pair_medium_over_hard_weight", 0.0)) * 0.25
    if right_tire == "SOFT" and left_tire == "HARD":
        pref += float(metadata.get("rerank_pair_soft_over_hard_weight", 0.0))
        if regime == "hot":
            pref += float(metadata.get("rerank_pair_hot_soft_over_hard_weight", 0.0))

    pref += float(metadata.get(f"rerank_pair_track_regime_{track}_{regime}_{pair_key}_weight", 0.0))

    if right_context["signature"] == left_context["signature"]:
        pref += float(metadata.get("rerank_same_strategy_bias", 0.0))
        pref += float(metadata.get("rerank_same_strategy_driver_weight", 0.0)) * (
            left_context["driver_rank_norm"] - right_context["driver_rank_norm"]
        )

    return pref


def _apply_local_rerank(scored, strategies, race_config, model, dynamic_overrides=None):
    metadata = model.get("metadata", {})
    if not metadata.get("rerank_enabled", False):
        return scored

    margin = float(metadata.get("rerank_margin", 0.0))
    if margin <= 0.0:
        return scored

    rounds = max(1, int(metadata.get("rerank_rounds", 1)))
    window = max(1, int(metadata.get("rerank_window", 2)))
    if dynamic_overrides:
        margin += float(dynamic_overrides.get("local_margin_boost", 0.0))
        rounds += int(dynamic_overrides.get("local_rounds_extra", 0))
        window += int(dynamic_overrides.get("local_window_extra", 0))
    by_driver = {strategy["driver_id"]: strategy for strategy in strategies.values()}
    contexts = {driver_id: _build_rerank_context(strategy, race_config) for driver_id, strategy in by_driver.items()}

    out = list(scored)
    for _ in range(rounds):
        changed = False
        for i in range(len(out) - 1):
            max_j = min(len(out), i + 1 + window)
            for j in range(i + 1, max_j):
                left_t, left_d = out[j - 1]
                right_t, right_d = out[j]
                gap = right_t - left_t
                if gap > margin:
                    break
                pref = _local_pair_preference(contexts[left_d], contexts[right_d], gap, metadata)
                if pref > 0.0:
                    out[j - 1], out[j] = out[j], out[j - 1]
                    changed = True
        if not changed:
            break
    return out


def simulate_race(race_config, strategies, model):
    scored = []
    for pos_key in sorted(strategies.keys(), key=lambda key: int(key[3:])):
        strategy = strategies[pos_key]
        driver_id = strategy["driver_id"]
        relative_time = _driver_relative_time(strategy, race_config, model)
        relative_time = _apply_runtime_adjustments(relative_time, strategy, race_config, model)
        scored.append((relative_time, driver_id))

    scored.sort(key=lambda item: item[0])

    # Deterministic near-tie resolver: apply only when adjacent totals are very close.
    metadata = model.get("metadata", {})
    tie_scores = metadata.get("tie_break_scores", {})
    gap_threshold = float(metadata.get("tie_gap_threshold", 0.0))
    if gap_threshold > 0.0 and tie_scores:
        changed = True
        while changed:
            changed = False
            for i in range(len(scored) - 1):
                t_left, d_left = scored[i]
                t_right, d_right = scored[i + 1]
                if (t_right - t_left) <= gap_threshold and tie_scores.get(d_right, 0.0) > tie_scores.get(d_left, 0.0):
                    scored[i], scored[i + 1] = scored[i + 1], scored[i]
                    changed = True

    dynamic_overrides = _dynamic_rerank_overrides(scored, metadata)

    # Apply learned pairwise reranker first, then metadata-driven local reranker.
    scored = _apply_pairwise_reranker(scored, strategies, race_config, model, dynamic_overrides)
    scored = _apply_local_rerank(scored, strategies, race_config, model, dynamic_overrides)

    return [driver_id for _, driver_id in scored]


def main():
    test_case = json.load(sys.stdin)
    race_id = test_case["race_id"]
    race_config = test_case["race_config"]
    strategies = test_case["strategies"]

    model = _load_model()
    finishing_positions = simulate_race(race_config, strategies, model)

    output = {
        "race_id": race_id,
        "finishing_positions": finishing_positions,
    }
    print(json.dumps(output))


if __name__ == "__main__":
    main()