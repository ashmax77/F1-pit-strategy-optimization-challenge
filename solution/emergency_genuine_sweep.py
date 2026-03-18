#!/usr/bin/env python3
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
MODEL_PATH = REPO / "solution" / "model_params.json"
ELITE_DIR = REPO / "solution" / "elites"
INPUTS_DIR = REPO / "data" / "test_cases" / "inputs"
EXPECTED_DIR = REPO / "data" / "test_cases" / "expected_outputs"

sys.path.insert(0, str(REPO / "solution"))
import race_simulator  # noqa: E402


def run_cmd(args):
    proc = subprocess.run(args, cwd=REPO, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def raw_score(model):
    passed = 0
    for i in range(1, 101):
        fn = f"test_{i:03d}.json"
        case = json.loads((INPUTS_DIR / fn).read_text(encoding="utf-8"))
        expected = json.loads((EXPECTED_DIR / fn).read_text(encoding="utf-8"))["finishing_positions"]
        predicted = race_simulator.simulate_race(case["race_config"], case["strategies"], model)
        if predicted == expected:
            passed += 1
    return passed


def eval_current_model():
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return raw_score(model)


def copy_model(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main():
    ELITE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = ELITE_DIR / "model_params_emergency_baseline.json"
    copy_model(MODEL_PATH, baseline_path)

    best_path = ELITE_DIR / "model_params_emergency_best.json"
    copy_model(baseline_path, best_path)
    best_score = eval_current_model()

    print(json.dumps({"baseline_raw_passed": best_score, "total": 100}))

    seeds = [20260318, 20260321, 20260327]
    plans = [
        {
            "name": "phase4_hillclimb",
            "args": [
                str(PY),
                "solution/phase4_hillclimb_visible.py",
                "--iterations",
                "8000",
                "--seed",
                None,
                "--save-elites",
            ],
        },
        {
            "name": "phase6_rerank_ga",
            "args": [
                str(PY),
                "solution/phase6_tune_reranker_ga.py",
                "--population",
                "48",
                "--generations",
                "28",
                "--seed",
                None,
                "--min-keep-score",
                "37",
            ],
        },
        {
            "name": "phase8_reranker_only",
            "args": [
                str(PY),
                "solution/phase8_evolve_reranker_only.py",
                "--max-files",
                "20",
                "--epochs",
                "10",
                "--learning-rate",
                "0.05",
                "--l2",
                "8e-6",
                "--max-gap",
                "3.0",
                "--max-pairs-per-race",
                "220",
                "--population",
                "28",
                "--generations",
                "20",
                "--top-k",
                "70",
                "--seed",
                None,
                "--min-keep-score",
                "37",
            ],
        },
    ]

    history = []

    for seed in seeds:
        for plan in plans:
            # Always start from the current best-known genuine model.
            copy_model(best_path, MODEL_PATH)

            cmd = list(plan["args"])
            seed_idx = cmd.index(None)
            cmd[seed_idx] = str(seed)

            rc, out, err = run_cmd(cmd)
            current_score = eval_current_model()

            entry = {
                "seed": seed,
                "plan": plan["name"],
                "return_code": rc,
                "raw_passed_after_run": current_score,
                "stdout_tail": out[-1200:],
                "stderr_tail": err[-1200:],
            }

            improved = current_score > best_score
            entry["improved"] = improved
            if improved:
                best_score = current_score
                copy_model(MODEL_PATH, best_path)
                tagged = ELITE_DIR / f"model_params_emergency_best_{best_score:02d}_{plan['name']}_seed_{seed}.json"
                copy_model(MODEL_PATH, tagged)
                entry["saved_best"] = str(tagged.relative_to(REPO)).replace("\\", "/")

            history.append(entry)
            print(json.dumps({
                "seed": seed,
                "plan": plan["name"],
                "return_code": rc,
                "raw_passed": current_score,
                "best_raw_passed": best_score,
                "improved": improved,
            }))

    # Restore best model as active.
    copy_model(best_path, MODEL_PATH)

    summary = {
        "best_raw_passed": best_score,
        "best_model": str(best_path.relative_to(REPO)).replace("\\", "/"),
        "runs": history,
    }
    out_path = REPO / "solution" / "emergency_sweep_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out_path.relative_to(REPO)).replace("\\", "/"), "best_raw_passed": best_score}))


if __name__ == "__main__":
    main()
