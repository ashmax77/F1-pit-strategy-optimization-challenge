#!/usr/bin/env python3
"""Deep diff of predictions vs expected for specific test cases.
Shows exact driver ordering differences and score gaps."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_simulator import _driver_relative_time, _load_model

repo = Path(__file__).resolve().parents[1]
inputs_dir = repo / "data" / "test_cases" / "inputs"
expected_dir = repo / "data" / "test_cases" / "expected_outputs"

# Target cases with fewest mismatches
targets = [9, 20, 24, 40, 53, 96, 22, 85, 33, 51, 56, 77]

model = _load_model()

for i in targets:
    fn = f"test_{i:03d}.json"
    case = json.loads((inputs_dir / fn).read_text(encoding="utf-8"))
    exp = json.loads((expected_dir / fn).read_text(encoding="utf-8"))["finishing_positions"]

    race_config = case["race_config"]
    strategies = case["strategies"]

    # Score every driver
    scored = []
    for pos_key in sorted(strategies.keys(), key=lambda k: int(k[3:])):
        s = strategies[pos_key]
        t = _driver_relative_time(s, race_config, model)
        scored.append((t, s["driver_id"]))
    scored.sort()
    pred = [d for _, d in scored]

    mismatches = sum(1 for a, b in zip(pred, exp) if a != b)
    track = race_config["track"]
    temp = race_config["track_temp"]

    print(f"\n{'='*60}")
    print(f"TEST_{i:03d} | {track} | temp={temp} | mismatches={mismatches}")
    print(f"{'Pos':>4}  {'Expected':>8}  {'Predicted':>9}  {'Match':>6}  {'Score':>10}")

    # Build score map
    score_map = {d: t for t, d in scored}

    for pos, (e, p) in enumerate(zip(exp, pred), 1):
        ok = "OK" if e == p else "WRONG"
        se = score_map.get(e, 0.0)
        print(f"{pos:>4}  {e:>8}  {p:>9}  {ok:>6}  {se:>10.4f}")

    # Show scores for expected order to see why they're wrong
    print(f"\n  Scores in EXPECTED order:")
    for pos, d in enumerate(exp, 1):
        s = score_map[d]
        p_pos = pred.index(d) + 1
        match = "OK" if pos == p_pos else f"->pos{p_pos}"
        pit_s = strategies.get(f"pos{pos}", {})
        # Find this driver's strategy
        drv_strat = None
        for ps, st in strategies.items():
            if st["driver_id"] == d:
                drv_strat = st
                break
        stops = drv_strat.get("pit_stops", []) if drv_strat else []
        last_tire = stops[-1]["to_tire"] if stops else (drv_strat["starting_tire"] if drv_strat else "?")
        stop_laps = [st["lap"] for st in stops]
        print(f"  {pos:>3}. {d} score={s:>10.4f} {match:>10}  final_tire={last_tire}  stops@{stop_laps}")
